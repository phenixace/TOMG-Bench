## MolCustom
Target: Let the LLM generate the customized molecule. Match if the generated molecule meets the standard of the requirements.

### Subtasks

#### AtomNum
- **Description**: The number of atoms in the generated molecule should be equal to the given number.   
- **Input**: The instruction that specifies the number of atoms in the generated molecule.
- **Output**: The molecule SMILES
- **Example**: 
  - Input: `Please generate a molecule with 8 carbon atoms, 1 nitrogen atoms, and 2 oxygen atoms.`
  - Output: `CCCCC(C)NCC(=O)O`
- **Evaluation Metrics**: 
  - **Success Rate (MAIN)**: The percentage of generated molecules that meet the requirements. 
  - **Molecule Novelty**: The percentage of generated molecules that are novel.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### BasicProp
- **Description**: The generated molecule should meet the basic properties, such as toxity, solubility, etc.
- **Input**: The instruction that specifies the basic properties of the generated molecule.
- **Output**: The molecule SMILES
- **Example**: 
  - Input: `Please generate a molecule with low toxicity.`
  - Output: `c1ccccc1O`
- **Evaluation Metrics**: 
  - **Success Rate (MAIN)**: The percentage of generated molecules that meet the requirements. We could apply GNN Models that have been trained on datasets with toxicity labels to predict the toxicity of the generated molecules.
  - **Molecule Novelty**: The percentage of generated molecules that are novel.
  - **Molecule Validity**: The percentage of generated molecules that are valid.

#### FunctionalGroup
- **Description**: The generated molecule should contain the specified functional groups.
- **Input**: The instruction that specifies the numbers of the functional groups in the generated molecule.
- **Output**: The molecule SMILES
- **Example**: 
  - Input: `Please generate a molecule with 2 hydroxyl groups.`
  - Output: `OCCCCO`
- **Evaluation Metrics**:
    - **Success Rate (MAIN)**: The percentage of generated molecules that meet the requirements. 
    - **Molecule Novelty**: The percentage of generated molecules that are novel.
    - **Molecule Validity**: The percentage of generated molecules that are valid.

#### BondNum
- **Description**: The generated molecule should contain the specified number of bonds.
- **Input**: The instruction that specifies the number of bonds in the generated molecule.
- **Output**: The molecule SMILES
- **Example**: 
  - Input: `Please generate a molecule with 1 single bond.`
  - Output: `CC`
- **Evaluation Metrics**:
    - **Success Rate (MAIN)**: The percentage of generated molecules that meet the requirements. 
    - **Molecule Novelty**: The percentage of generated molecules that are novel.
    - **Molecule Validity**: The percentage of generated molecules that are valid.