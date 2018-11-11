from IALS import IALS
import numpy as np
from datetime import datetime

ials1 = IALS(max_epoch=35,
            embedding_size=6,
            alpha=35,
            l2reg=7.5,
            eps=.11,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=3,
            mean_decrease=0.85)

ials2 = IALS(max_epoch=35,
            embedding_size=15,
            alpha=35,
            l2reg=9.5,
            eps=.11,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=3,
            mean_decrease=0.85)

ials3 = IALS(max_epoch=40,
            embedding_size=4,
            alpha=25,
            l2reg=2,
            eps=.07,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=3,
            mean_decrease=0.85)
R = np.load("data/train.npy")
R_test = np.load("data/test.npy")
start_time = datetime.now()
res1, _, _ = ials1.fit(R, R_test)
res2, _, _ = ials2.fit(R, R_test)
res3, _, _ = ials3.fit(R, R_test)
res = (res1 + res2 + res3) / 3
submission_name = ("data/submission_{}.txt".format(datetime.now())).replace(' ', '_').replace(":","_")
with open("data/test.txt", 'r') as file_in:
    with open(submission_name, 'w') as file_out:
        file_out.write("Id,Score\n")
        for id, line in enumerate(file_in):
            ui = [int(a) for a in line.strip().split('\t')]
            assert len(ui) == 2
            file_out.write("{},{}\n".format(id + 1, res[ui[0] - 1, ui[1] - 1]))
print("Time elapsed: {}".format(datetime.now() - start_time))
