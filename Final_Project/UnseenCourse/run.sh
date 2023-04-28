# "${1}": path to the users file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.
python3.9 best_seller.py --do_predict --users_file "${1}" --test_file "${2}" --test_pred_file "${3}"