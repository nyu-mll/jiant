target=${1:-/scratch/zp489/share/nlutransfer/}
recurse_mode=${2:-recurse}
user_list=(sb6065 kk120 hl3236 yp883 zp489 yp913 asw462 xz2448 pmh330 cv50)

if [[ $recurse_mode == "recurse" ]]; then
    for user in ${user_list[*]}; do
        echo "Recursively giving permissions to ${target} to ${user}"
        setfacl -R -m u:${user}:rwx $target
    done
elif [[ $recurse_mode == "norecurse" ]]; then
    for user in ${user_list[*]}; do
        echo "Giving permissions to ${target} to ${user}"
        setfacl -m u:${user}:rwx ${target}
    done
else
    echo "options: recurse|norecurse"
fi
