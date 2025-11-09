"""
Integration module: RAG System + Nodal Analysis
Connects ChromaDB + Ollama with nodal analysis calculations

Usage:
    python examples/sub_challenge_3.py --well "NLW-GT-03"
"""

import chromadb
from chromadb.config import Settings
import ollama
from pathlib import Path
from typing import Dict, List, Optional
import json
import re
from sentence_transformers import SentenceTransformer


class DocumentProcessor:
    """Process and index well documents into vector store"""
    
    def __init__(self, collection_name: str = "well_reports"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Well completion reports"}
        )
    
    def index_documents(self, well_folder: Path) -> None:
        """
        Index all documents from a well folder into ChromaDB
        
        Args:
            well_folder: Path to well documents folder
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        
        # Process all PDFs
        for pdf_file in well_folder.glob("*.pdf"):
            print(f"Processing: {pdf_file.name}")
            
            # Convert PDF to text with OCR
            result = converter.convert(str(pdf_file))
            text = result.document.export_to_markdown()
            
            # Chunk text
            chunks = self._chunk_text(text, chunk_size=500)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Add to ChromaDB
            ids = [f"{pdf_file.stem}_chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"source": pdf_file.name} for _ in chunks]
            )
        
        print(f"✓ Indexed {self.collection.count()} chunks")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, 
                    overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def query(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Query the vector store for relevant chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i]
            })
        
        return formatted


class LLMParameterExtractor:
    """Extract structured parameters using Ollama + Llama 3.2"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.client = ollama.Client()
    
    def extract_parameters(self, query: str, context: List[Dict]) -> Dict:
        """
        Extract parameters from retrieved context using LLM
        
        Args:
            query: Parameter extraction query
            context: List of retrieved document chunks
            
        Returns:
            Dictionary of extracted parameters
        """
        # Format context
        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}"
            for c in context
        ])
        
        # Create extraction prompt
        prompt = f"""You are an expert in petroleum engineering analyzing well completion reports.

Your task: Extract the following parameters from the well report excerpts below.

Query: {query}

Well Report Excerpts:
{context_text}

Instructions:
1. Carefully read all excerpts
2. Extract ONLY the requested parameters with their numerical values and units
3. If a parameter is not found, return null
4. Return your answer as a JSON object
5. Do not include any preamble or markdown formatting

Respond with ONLY valid JSON in this format:
{{
    "parameter_name": {{"value": number, "unit": "string"}},
    ...
}}"""

        # Call Ollama
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
            }
        )
        
        # Parse response
        return self._parse_response(response['response'])
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*|\s*```', '', response)
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response}")
            return {}


class RAGNodalAnalysisAgent:
    """
    Main agent that orchestrates the entire workflow:
    1. Index documents
    2. Extract parameters via RAG
    3. Perform nodal analysis
    4. Generate report
    """
    
    def __init__(self, well_name: str, data_folder: Path):
        self.well_name = well_name
        self.data_folder = data_folder
        self.well_folder = data_folder / f"Well {well_name}"
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.llm_extractor = LLMParameterExtractor()
    
    def run(self) -> Dict:
        """Execute complete RAG-Nodal analysis workflow"""
        
        print(f"\n{'='*70}")
        print(f"RAG-Nodal Analysis Agent: {self.well_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Index documents
        print("Step 1: Indexing well documents...")
        if not self.well_folder.exists():
            raise FileNotFoundError(f"Well folder not found: {self.well_folder}")
        
        self.doc_processor.index_documents(self.well_folder)
        print()
        
        # Step 2: Extract parameters using RAG
        print("Step 2: Extracting parameters via RAG...")
        parameters = self._extract_all_parameters()
        print("  ✓ Parameter extraction complete")
        print()
        
        # Step 3: Perform nodal analysis
        print("Step 3: Performing nodal analysis...")
        from nodal_analysis_rag import WellParameters, NodalAnalysis
        
        well_params = WellParameters(**parameters)
        nodal = NodalAnalysis(well_params)
        results = nodal.estimate_production_capacity()
        print("  ✓ Nodal analysis complete")
        print()
        
        # Step 4: Generate report
        print("Step 4: Generating report...")
        report = self._generate_report(parameters, results)
        print()
        
        return report
    
    def _extract_all_parameters(self) -> Dict:
        """Extract all required parameters using RAG"""
        
        # Define parameter queries
        queries = {
            "geometry": {
                "query": "What are the measured depth (MD), true vertical depth (TVD), and inner diameter (ID) or tubing size of the well? Include numerical values with units.",
                "params": ["measured_depth", "true_vertical_depth", "inner_diameter"]
            },
            "reservoir": {
                "query": "What are the reservoir pressure, reservoir temperature, permeability, and skin factor? Include numerical values with units.",
                "params": ["reservoir_pressure", "reservoir_temperature", "permeability", "skin_factor"]
            },
            "fluid": {
                "query": "What are the oil API gravity, gas specific gravity, water cut percentage, and gas-oil ratio (GOR)? Include numerical values.",
                "params": ["oil_api_gravity", "gas_specific_gravity", "water_cut", "gor"]
            },
            "production": {
                "query": "What are the wellhead pressure and separator pressure? Include numerical values with units.",
                "params": ["wellhead_pressure", "separator_pressure"]
            }
        }
        
        extracted_params = {}
        
        for category, info in queries.items():
            print(f"  Extracting {category} parameters...")
            
            # Retrieve relevant chunks
            chunks = self.doc_processor.query(info["query"], n_results=5)
            
            # Extract parameters using LLM
            params = self.llm_extractor.extract_parameters(info["query"], chunks)
            
            # Convert to standard format
            for param_name in info["params"]:
                if param_name in params and params[param_name]:
                    value = params[param_name].get("value")
                    if value is not None:
                        extracted_params[param_name] = float(value)
        
        # Fill in defaults for missing parameters
        defaults = {
            "measured_depth": 3500.0,
            "true_vertical_depth": 3450.0,
            "inner_diameter": 5.5,
            "reservoir_pressure": 4500.0,
            "reservoir_temperature": 350.0,
            "permeability": 50.0,
            "skin_factor": 2.5,
            "oil_api_gravity": 35.0,
            "gas_specific_gravity": 0.65,
            "water_cut": 0.2,
            "gor": 500.0,
            "wellhead_pressure": 200.0,
            "separator_pressure": 100.0
        }
        
        for param, default_value in defaults.items():
            if param not in extracted_params:
                print(f"    ⚠ {param} not found, using default: {default_value}")
                extracted_params[param] = default_value
        
        return extracted_params
    
    def _generate_report(self, parameters: Dict, results: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        
        report = {
            "well_name": self.well_name,
            "analysis_date": "2025-01-04",
            "parameters": parameters,
            "results": results,
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report
        output_file = f"outputs/nodal_analysis_{self.well_name}.json"
        Path("outputs").mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(report, indent=2, fp=f)
        
        print(f"  ✓ Report saved to: {output_file}")
        
        # Display summary
        self._display_summary(report)
        
        return report
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        op = results["operating_point"]
        metrics = results["metrics"]
        
        # Check productivity
        if metrics["productivity_index"] < 1.0:
            recommendations.append(
                "Low productivity index suggests formation damage. Consider stimulation treatment."
            )
        
        # Check water cut
        if metrics["water_cut"] > 50:
            recommendations.append(
                "High water cut detected. Evaluate water shut-off techniques."
            )
        
        # Check drawdown
        if metrics["drawdown"] > 1000:
            recommendations.append(
                "High drawdown pressure. Consider artificial lift installation."
            )
        
        if not recommendations:
            recommendations.append("Well is operating within normal parameters.")
        
        return recommendations
    
    def _display_summary(self, report: Dict):
        """Display report summary"""
        
        print(f"\n{'='*70}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*70}\n")
        
        op = report["results"]["operating_point"]
        metrics = report["results"]["metrics"]
        
        print(f"Well: {report['well_name']}")
        print()
        print("Production Capacity:")
        print(f"  Total Liquid: {op['total_liquid_rate']} bbl/day")
        print(f"  Oil Rate:     {op['oil_rate']} bbl/day")
        print(f"  Gas Rate:     {op['gas_rate']} scf/day")
        print()
        print("Key Metrics:")
        print(f"  Productivity Index: {metrics['productivity_index']} bbl/day/psi")
        print(f"  Water Cut:          {metrics['water_cut']}%")
        print()
        print("Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        print()
        print(f"{'='*70}\n")


def main():
    """Main entry point for Sub-Challenge 3"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG-powered nodal analysis for well production estimation"
    )
    parser.add_argument(
        "--well",
        type=str,
        default="NLW-GT-03",
        help="Well name (e.g., NLW-GT-03)"
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path("Training data-shared with participants"),
        help="Path to training data folder"
    )
    
    args = parser.parse_args()
    
    # Run agent
    agent = RAGNodalAnalysisAgent(args.well, args.data_folder)
    report = agent.run()
    
    return report


if __name__ == "__main__":
    main()
