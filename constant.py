
"""
the features refer to a set of pairs of face landmarks point  
"""

FEATURE_PART_NUMBER = [
[18,22],[37,40],[39,41],[39,21],[40,22],        #left eye
[23,27],[43,46],[44,48],[44,24],[43,23],        #right eye
[22,23],[40,43],                                #top-central-face
[37,49],[46,55],                                #link-eyes-blow
[40,32],[36,43],                                #link-eyes-nose
[32,49],[36,55],[34,52],                        #link-nose-blow
[52,58],[63,67],[49,55],                        #blow-important
[49,52],[52,55],[55,58],[58,49],[52,63],[63,58] #blow-prove
]


FEATURE_PART_NUMBER_2 = [
[18,22],[18,37],[22,40],[37,36],[36,40],[40,42],[42,37],[40,34],[34,32],[32,40],[34,49],[49,42],[49,58],
[23,27],[23,43],[43,45],[45,46],[46,47],[47,43],[27,46],[47,55],[43,34],[43,36],[36,34],[34,55],[55,58]
]

"""
The paths used to train,test and retrieving information from the model
"""

PATH_NAME_TO_TEST_SET = "dataset.test"
PATH_NAME_TO_TRAIN_SET = "dataset.train"
PATH_NAME_TO_MODEL = "model.sav"