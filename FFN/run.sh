


batchsize_params=(
    "8"
    "16"
    "32"
    "64"
    "128"
    "256"
)

drop_params=(
    "0.1"
    "0.2"
    "0.3"
    "0.4"
    "0.5"
)


for param in "${batchsize_params[@]}"
do
    echo "Running  with parameter: batchsize:$params, "
    python ./train.py --batchsize $params >> ./log/train.log 2>&1
done
