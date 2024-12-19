## MolEdit
Target: Let the LLM edit the molecule and optimize the molecule with specific properties. Match if the generated molecule meets the standard of the requirements.

### Subtasks

#### LogP
- **Description**: Optimize the molecule to have the higher or lower LogP value.
- **Input**: The instruction that specifies the target LogP value direction.
- **Output**: The optimized molecule SMILES
- **Example**: 
  - Input: `Please optimize the molecule CCO to have a higher LogP value.`
  - Output: `OCCO`
- **Evaluation Metrics**: 
  - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
  - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### MR
- **Description**: Optimize the molecule to have the higher or lower MR value.
- **Input**: The instruction that specifies the target MR value direction.
- **Output**: The optimized molecule SMILES
- **Example**: 
  - Input: `Please optimize the molecule CCO to have a higher MR value.`
  - Output: `CCCC`
- **Evaluation Metrics**: 
  - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
  - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### Toxity
- **Description**: Optimize the molecule to have the higher or lower toxicity.
- **Input**: The instruction that specifies the target toxicity direction.
- **Output**: The optimized molecule SMILES
- **Example**: 
  - Input: `Please optimize the molecule CCO to have a higher toxicity.`
  - Output: `CCN`
- **Evaluation Metrics**:
    - **Success Rate (Accuracy) (MAIN)**: The percentage of generated molecules that meet the requirements. 
    - **Molecule Similarity**: The edited molecule should make as few changes as possible to the original molecule.
    - **Molecule Validity**: The percentage of generated molecules that are valid.