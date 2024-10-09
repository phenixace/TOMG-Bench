分子描述符计算
计算分子描述符：如分子量、LogP、极性表面积等。
示例：
```
python
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles('CCO')
mol_weight = Descriptors.MolWt(mol)  # 计算分子量
logp = Descriptors.MolLogP(mol)  # 计算LogP
```