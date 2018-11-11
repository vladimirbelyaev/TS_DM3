import numpy as np
from IALS import IALS, train_test_split
from datetime import datetime
R = np.load("data/train.npy")
R_train, R_test = train_test_split(R, 0.1, 1234)
'''ials = IALS(max_epoch=300,
            embedding_size=4,
            alpha=5,
            verbose=25,
            l2reg=0.07,
            eps=0.1,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            mean_decrease=0.85)
'''

'''
ials = IALS(max_epoch=300,
            embedding_size=7,
            alpha=25,
            verbose=5,
            l2reg=10,
            eps=.1,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            mean_decrease=0.85)
            '''

'''
ials = IALS(max_epoch=100,
            embedding_size=3,
            alpha=5,
            l2reg=0.1,
            eps=.11,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=5,
            mean_decrease=0.85)'''
'''
ials = IALS(max_epoch=120,
            embedding_size=9,
            alpha=35,
            l2reg=7,
            eps=.1,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=5,
            mean_decrease=0.85)
            '''
#best - не трогать
'''
ials = IALS(max_epoch=35,
            embedding_size=6,
            alpha=35,
            l2reg=7.5,
            eps=.11,
            use_test_to_init=True,
            log_confidence=True,
            show_real_metric=False,
            verbose=3,
            mean_decrease=0.85)
            '''

'''
ials = IALS(max_epoch=150,
            embedding_size=5,
            alpha=5,
            l2reg=0.03,
            eps=.11,
            use_test_to_init=True,
            #confidence="log",
            log_confidence=True,
            show_real_metric=False,
            verbose=3,
            mean_decrease=0.85,
            normalisation="+bias"
            )'''

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




start_time = datetime.now()
#res = ials.fit(R_train, R_test)

res1 = ials1.fit(R_train, R_test)
res2 = ials2.fit(R_train, R_test)
res3 = ials3.fit(R_train, R_test)
err = ((((res1[0] + res2[0] + res3[0]) / 3)[R_test > 0] - R_test[R_test > 0])**2).sum() / (R_test > 0).sum()
print(np.sqrt(err))
print("Time elapsed: {}".format(datetime.now() - start_time))