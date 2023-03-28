cfg_name="swin_tiny"

CUDA_VISIBLE_DEVICES=0 python ./tools/test_city_person.py configs/solider/cp/${cfg_name}.py ./work_dirs/cp/$cfg_name/epoch_ 1 241 --out ${cfg_name}.json --mean_teacher 2>&1 | tee ${cfg_name}.txt
