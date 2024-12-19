## MolEdit
Target: Let the LLM edit the molecule. Match if the generated molecule meets the standard of the requirements.

### Subtasks

#### AddComponent
- **Description**: Add the specified functional groups to the molecule.
- **Input**: The instruction that specifies the functional groups to be added to the molecule.
- **Output**: The edited molecule SMILES
- **Example**: 
  - Input: `Please add a hydroxyl group to the molecule CC.`
  - Output: `CCO`
- **Evaluation Metrics**: 
  - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
  - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### RemoveComponent (Is this open? Could be open if the instruction is not specific (number/group))
- **Description**: Remove the specified functional groups from the molecule.
- **Input**: The instruction that specifies the functional groups to be removed from the molecule.
- **Output**: The edited molecule SMILES
- **Example**: 
  - Input: `Please remove a hydroxyl group from the molecule CCO.`
  - Output: `CC`
- **Evaluation Metrics**: 
  - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
  - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### SubstituteComponent (Is this open? Could be open if we do not specify which functional group to substitute)
- **Description**: Substitute the specified functional groups in the molecule.
- **Input**: The instruction that specifies the functional groups to be substituted in the molecule.
- **Output**: The edited molecule SMILES
- **Example**: 
  - Input: `Please substitute a hydroxyl group in the molecule CCO with a carboxyl group.`
  - Output: `CCN`
- **Evaluation Metrics**:
    - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
    - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
    - **Molecule Validity**: The percentage of generated molecules that are valid.