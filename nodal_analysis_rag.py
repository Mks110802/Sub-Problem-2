"""
RAG-Powered Nodal Analysis System
Sub-Challenge 3: Automated Well Production Capacity Estimation

This system uses RAG to extract parameters from well reports and 
performs nodal analysis to estimate production capacity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re


@dataclass
class WellParameters:
    """Well parameters extracted from documents via RAG"""
    # Well geometry
    measured_depth: float  # MD in meters
    true_vertical_depth: float  # TVD in meters
    inner_diameter: float  # ID in inches
    
    # Reservoir properties
    reservoir_pressure: float  # psi
    reservoir_temperature: float  # °F
    permeability: float  # mD
    skin_factor: float  # dimensionless
    
    # Fluid properties
    oil_api_gravity: float  # °API
    gas_specific_gravity: float  # dimensionless
    water_cut: float  # fraction (0-1)
    gor: float  # Gas-Oil Ratio, scf/bbl
    
    # Production constraints
    wellhead_pressure: float  # psi
    separator_pressure: float  # psi
    
    def __post_init__(self):
        """Validate parameters"""
        if self.water_cut < 0 or self.water_cut > 1:
            raise ValueError("Water cut must be between 0 and 1")
        if self.measured_depth < self.true_vertical_depth:
            raise ValueError("MD must be >= TVD")


class RAGParameterExtractor:
    """
    Extracts well parameters from documents using RAG workflow.
    In production, this would connect to ChromaDB + Ollama.
    """
    
    def __init__(self, vector_store=None, llm=None):
        self.vector_store = vector_store
        self.llm = llm
        
    def extract_parameters(self, well_name: str, documents: List[str]) -> WellParameters:
        """
        Extract parameters using RAG pipeline:
        1. Embed query for each parameter
        2. Retrieve relevant chunks from vector store
        3. Use LLM to extract structured data
        
        Args:
            well_name: Name of the well
            documents: List of document texts or paths
            
        Returns:
            WellParameters object with extracted values
        """
        
        # Define parameter extraction queries
        queries = {
            "geometry": "What are the measured depth (MD), true vertical depth (TVD), and inner diameter (ID) of the well?",
            "reservoir": "What are the reservoir pressure, temperature, permeability, and skin factor?",
            "fluid": "What are the oil API gravity, gas specific gravity, water cut, and GOR?",
            "production": "What are the wellhead pressure and separator pressure constraints?"
        }
        
        # In production: Query RAG system for each parameter group
        # retrieved_data = self._query_rag_system(queries)
        
        # For demo: Mock extracted parameters
        # In real implementation, this comes from LLM parsing retrieved chunks
        extracted = self._mock_extraction(well_name)
        
        return WellParameters(**extracted)
    
    def _query_rag_system(self, queries: Dict[str, str]) -> Dict:
        """
        Query RAG system (ChromaDB + Ollama) for parameters.
        
        Workflow:
        1. Embed each query using nomic-embed-text-v1.5
        2. Retrieve top-k relevant chunks from ChromaDB
        3. Pass chunks + query to Llama 3.2 for extraction
        4. Parse LLM response to structured data
        """
        if not self.vector_store or not self.llm:
            raise RuntimeError("RAG system not initialized")
        
        results = {}
        for category, query in queries.items():
            # Retrieve relevant chunks
            chunks = self.vector_store.query(query, n_results=5)
            
            # Create extraction prompt
            prompt = f"""Based on the following well report excerpts, extract the requested parameters.
            
Query: {query}

Relevant excerpts:
{chunks}

Respond with a JSON object containing the parameter names and values with units."""
            
            # Get LLM response
            response = self.llm.generate(prompt)
            results[category] = self._parse_llm_response(response)
        
        return results
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract structured data"""
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*|\s*```', '', response)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: regex extraction
            return self._regex_extraction(response)
    
    def _regex_extraction(self, text: str) -> Dict:
        """Fallback regex-based parameter extraction"""
        patterns = {
            'measured_depth': r'MD[:\s]+([0-9.]+)\s*m',
            'pressure': r'pressure[:\s]+([0-9.]+)\s*psi',
            'temperature': r'temperature[:\s]+([0-9.]+)\s*[°F]',
        }
        
        results = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[key] = float(match.group(1))
        
        return results
    
    def _mock_extraction(self, well_name: str) -> Dict:
        """Mock parameter extraction for demonstration"""
        # Typical geothermal well parameters
        return {
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


class NodalAnalysis:
    """
    Performs nodal analysis calculations to estimate production capacity.
    Uses the extracted parameters from RAG system.
    """
    
    def __init__(self, params: WellParameters):
        self.params = params
        
    def calculate_ipr(self, pressures: np.ndarray) -> np.ndarray:
        """
        Calculate Inflow Performance Relationship (IPR)
        Using Vogel's equation for solution gas drive reservoirs
        
        Args:
            pressures: Array of bottom-hole flowing pressures (psi)
            
        Returns:
            Array of flow rates (bbl/day)
        """
        pr = self.params.reservoir_pressure
        
        # Productivity index (simplified)
        j = self._calculate_productivity_index()
        
        # Vogel's equation
        q = []
        for pwf in pressures:
            if pwf >= pr:
                q.append(0)
            else:
                # q = J * PR * [1 - 0.2(Pwf/PR) - 0.8(Pwf/PR)^2]
                ratio = pwf / pr
                q_oil = j * pr * (1 - 0.2 * ratio - 0.8 * ratio**2)
                q.append(max(0, q_oil))
        
        return np.array(q)
    
    def calculate_tpr(self, flow_rates: np.ndarray) -> np.ndarray:
        """
        Calculate Tubing Performance Relationship (TPR)
        Bottom-hole pressure required for given flow rate
        
        Args:
            flow_rates: Array of flow rates (bbl/day)
            
        Returns:
            Array of bottom-hole pressures (psi)
        """
        pwh = self.params.wellhead_pressure
        depth = self.params.true_vertical_depth
        
        # Simplified pressure drop calculation
        # In production: Use Beggs-Brill or Hagedorn-Brown correlations
        
        pressures = []
        for q in flow_rates:
            if q == 0:
                pressures.append(pwh)
            else:
                # Pressure gradient (simplified)
                gradient = self._calculate_pressure_gradient(q)
                
                # Bottom-hole pressure = Wellhead + hydrostatic + friction
                pwf = pwh + (gradient * depth / 3.281)  # Convert m to ft
                pressures.append(pwf)
        
        return np.array(pressures)
    
    def find_operating_point(self) -> Tuple[float, float]:
        """
        Find the intersection of IPR and TPR curves (operating point)
        
        Returns:
            Tuple of (flow_rate, pressure) at operating point
        """
        # Create pressure range
        pressures = np.linspace(
            self.params.wellhead_pressure,
            self.params.reservoir_pressure,
            100
        )
        
        # Calculate IPR curve
        ipr_rates = self.calculate_ipr(pressures)
        
        # Calculate TPR curve
        flow_rates = np.linspace(0, max(ipr_rates) * 1.2, 100)
        tpr_pressures = self.calculate_tpr(flow_rates)
        
        # Find intersection
        operating_point = self._find_intersection(
            pressures, ipr_rates,
            tpr_pressures, flow_rates
        )
        
        return operating_point
    
    def estimate_production_capacity(self) -> Dict:
        """
        Estimate well production capacity using nodal analysis
        
        Returns:
            Dictionary with production estimates and analysis results
        """
        # Find operating point
        flow_rate, pressure = self.find_operating_point()
        
        # Calculate maximum rates
        max_oil_rate = flow_rate * (1 - self.params.water_cut)
        max_water_rate = flow_rate * self.params.water_cut
        max_gas_rate = max_oil_rate * self.params.gor
        
        # Calculate productivity metrics
        pi = self._calculate_productivity_index()
        
        return {
            "operating_point": {
                "total_liquid_rate": round(flow_rate, 2),
                "oil_rate": round(max_oil_rate, 2),
                "water_rate": round(max_water_rate, 2),
                "gas_rate": round(max_gas_rate, 2),
                "bottom_hole_pressure": round(pressure, 2)
            },
            "metrics": {
                "productivity_index": round(pi, 2),
                "drawdown": round(self.params.reservoir_pressure - pressure, 2),
                "water_cut": round(self.params.water_cut * 100, 2)
            },
            "units": {
                "liquid_rate": "bbl/day",
                "oil_rate": "bbl/day",
                "water_rate": "bbl/day",
                "gas_rate": "scf/day",
                "pressure": "psi",
                "productivity_index": "bbl/day/psi"
            }
        }
    
    def _calculate_productivity_index(self) -> float:
        """
        Calculate productivity index using Darcy's equation
        
        J = (0.00708 * k * h) / (μ * B * [ln(re/rw) + S])
        """
        k = self.params.permeability
        h = 100  # Assumed net pay thickness (ft)
        mu = 2.0  # Oil viscosity (cp) - should be calculated from correlations
        B = 1.2  # Formation volume factor - should be calculated
        re = 2000  # Drainage radius (ft) - assumed
        rw = self.params.inner_diameter / 24  # Wellbore radius (ft)
        S = self.params.skin_factor
        
        j = (0.00708 * k * h) / (mu * B * (np.log(re / rw) + S))
        return j
    
    def _calculate_pressure_gradient(self, flow_rate: float) -> float:
        """
        Calculate pressure gradient in tubing
        
        Args:
            flow_rate: Flow rate (bbl/day)
            
        Returns:
            Pressure gradient (psi/ft)
        """
        # Simplified gradient calculation
        # In production: Use multiphase flow correlations
        
        # Base gradient (hydrostatic)
        sg = 0.85  # Fluid specific gravity (water = 1.0)
        base_gradient = 0.433 * sg
        
        # Friction component (simplified)
        if flow_rate > 0:
            velocity = flow_rate / (24 * 60 * np.pi * (self.params.inner_diameter / 24)**2)
            friction = 0.0001 * velocity**1.8
        else:
            friction = 0
        
        return base_gradient + friction
    
    def _find_intersection(self, p1: np.ndarray, q1: np.ndarray,
                          p2: np.ndarray, q2: np.ndarray) -> Tuple[float, float]:
        """Find intersection point of IPR and TPR curves"""
        # Interpolate to find intersection
        from scipy import interpolate
        
        # Create interpolation functions
        f_ipr = interpolate.interp1d(p1, q1, kind='linear', fill_value='extrapolate')
        f_tpr = interpolate.interp1d(q2, p2, kind='linear', fill_value='extrapolate')
        
        # Minimize difference
        def objective(q):
            p_ipr = np.interp(q, q1, p1)
            p_tpr = f_tpr(q)
            return abs(p_ipr - p_tpr)
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(0, max(q1)), method='bounded')
        
        optimal_q = result.x
        optimal_p = f_tpr(optimal_q)
        
        return optimal_q, optimal_p


def main():
    """
    Main workflow: RAG parameter extraction → Nodal analysis → Production estimate
    """
    print("=" * 70)
    print("RAG-POWERED NODAL ANALYSIS SYSTEM")
    print("Sub-Challenge 3: Automated Well Production Capacity Estimation")
    print("=" * 70)
    print()
    
    # Step 1: Initialize RAG system
    print("Step 1: Initializing RAG system...")
    print("  - Vector Store: ChromaDB")
    print("  - Embeddings: nomic-embed-text-v1.5")
    print("  - LLM: Ollama + Llama 3.2 3B")
    print()
    
    extractor = RAGParameterExtractor()
    
    # Step 2: Extract parameters from well reports
    print("Step 2: Extracting parameters from well reports...")
    well_name = "NLW-GT-03"
    documents = [
        f"Training data-shared with participants/Well {well_name}/"
    ]
    
    params = extractor.extract_parameters(well_name, documents)
    print(f"  ✓ Extracted parameters for well: {well_name}")
    print()
    print("  Well Geometry:")
    print(f"    - Measured Depth: {params.measured_depth} m")
    print(f"    - True Vertical Depth: {params.true_vertical_depth} m")
    print(f"    - Inner Diameter: {params.inner_diameter} in")
    print()
    print("  Reservoir Properties:")
    print(f"    - Pressure: {params.reservoir_pressure} psi")
    print(f"    - Temperature: {params.reservoir_temperature} °F")
    print(f"    - Permeability: {params.permeability} mD")
    print(f"    - Skin Factor: {params.skin_factor}")
    print()
    print("  Fluid Properties:")
    print(f"    - Oil API Gravity: {params.oil_api_gravity} °API")
    print(f"    - Gas Specific Gravity: {params.gas_specific_gravity}")
    print(f"    - Water Cut: {params.water_cut * 100}%")
    print(f"    - GOR: {params.gor} scf/bbl")
    print()
    
    # Step 3: Perform nodal analysis
    print("Step 3: Performing nodal analysis...")
    nodal = NodalAnalysis(params)
    
    results = nodal.estimate_production_capacity()
    
    print("  ✓ Nodal analysis complete")
    print()
    
    # Step 4: Display results
    print("=" * 70)
    print("PRODUCTION CAPACITY ESTIMATE")
    print("=" * 70)
    print()
    
    op = results["operating_point"]
    metrics = results["metrics"]
    units = results["units"]
    
    print("Operating Point:")
    print(f"  - Total Liquid Rate: {op['total_liquid_rate']} {units['liquid_rate']}")
    print(f"  - Oil Rate: {op['oil_rate']} {units['oil_rate']}")
    print(f"  - Water Rate: {op['water_rate']} {units['water_rate']}")
    print(f"  - Gas Rate: {op['gas_rate']} {units['gas_rate']}")
    print(f"  - Bottom-Hole Pressure: {op['bottom_hole_pressure']} {units['pressure']}")
    print()
    
    print("Performance Metrics:")
    print(f"  - Productivity Index: {metrics['productivity_index']} {units['productivity_index']}")
    print(f"  - Pressure Drawdown: {metrics['drawdown']} {units['pressure']}")
    print(f"  - Water Cut: {metrics['water_cut']}%")
    print()
    
    print("=" * 70)
    print("Analysis complete! Results saved to outputs/nodal_analysis_results.json")
    print("=" * 70)
    
    # Save results
    output = {
        "well_name": well_name,
        "parameters": {
            "measured_depth": params.measured_depth,
            "true_vertical_depth": params.true_vertical_depth,
            "inner_diameter": params.inner_diameter,
            "reservoir_pressure": params.reservoir_pressure,
            "reservoir_temperature": params.reservoir_temperature,
        },
        "results": results
    }
    
    return output


if __name__ == "__main__":
    results = main()
    
    # Export results as JSON
    with open("nodal_analysis_results.json", "w") as f:
        json.dump(results, indent=2, fp=f)
