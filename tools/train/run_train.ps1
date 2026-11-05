# Training PC launcher (run via the scheduled task "pedestrian_train" so it survives ssh logout).
param($Data, $Name, $Model = 'yolo26n_person_init.pt', $Attention = 'sigmoid', $Epochs = 100, $Batch = 32, $Fraction = 1.0)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\$Name.log"
& D:\bitpose_train\venv\Scripts\python.exe train_person.py $Data --model $Model --attention $Attention --epochs $Epochs --batch $Batch --name $Name --fraction $Fraction *>> $log
"TRAIN_EXIT $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
