from typing import Optional, List, Tuple, Dict

from pydantic import BaseModel, Field

# Your Pydantic model
class DiagnosisOutput(BaseModel):
    analysis: str = Field(description="Your analysis of the current findings.")
    next_components: List[Tuple[List[str], List[str]]] = Field(
        description="Pairs of module paths in pair-wise [HF, vLLM] format to compare next."
    )
    suspected_root_cause: Optional[str] = Field(
        default=None, description="Optional: If a root cause has been identified"
    )
    reasoning: str = Field(description="Why you chose the next components.")
    complete: bool = Field(description="Whether the diagnosis is complete.")
    

class LogitCollectorOutput(BaseModel):
    code: str = Field(description="The valid Python code to be executed to collect the logits.")
    
class ComponentMappingOutput(BaseModel):
    component_mapping: List[Tuple[List[str], List[str]]] = Field(
        description="Pairs of module paths in pair-wise [HF, vLLM] format to compare next."
    )
    variables: Dict[str, List[str]] = Field(
        description="Variables used in the component mapping and the values they can take."
    )
    reasoning: Dict[str, Dict[str, str]] = Field(description="The reasoning for why the remaining components that are not in the component_mapping are not mapped. reasoning['HF'] and reasoning['vLLM'] are the reasoning for why the remaining components that are not in the component_mapping are not mapped in HF and vLLM respectively.")

class AlignActivationOutput(BaseModel):
    code: str = Field(description="The valid Python code to be executed to align activations.")
    reasoning: Dict[str, str] = Field(description="The reasoning for how an output variable in vLLM is aligned with activations in HuggingFace.")
    
class FinalAnalysisOutput(BaseModel):
    suspected_components: List[str] = Field(description="The components that are suspected to be the root cause of the issue.")
    reasoning: Dict[str, str] = Field(description="The reasoning for why the components are suspected to be the root cause of the issue, key is the component name in suspected_components and value is the reasoning for why the component is suspected to be the root cause of the issue.")