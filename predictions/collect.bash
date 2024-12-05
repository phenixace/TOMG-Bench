name="llama3.2-1B"
data_scale="large"

for task in 'MolCustom'
do
    for subtask in 'AtomNum' 'BondNum' 'FunctionalGroup'
    do
        python collect.py --name $name --task $task --subtask $subtask --data_scale $data_scale
    done
done

for task in 'MolEdit'
do
    for subtask in 'AddComponent' 'DelComponent' 'SubComponent'
    do
        python collect.py --name $name --task $task --subtask $subtask --data_scale $data_scale
    done
done

for task in 'MolOpt'
do
    for subtask in 'LogP' 'QED' 'MR'
    do
        python collect.py --name $name --task $task --subtask $subtask --data_scale $data_scale
    done
done