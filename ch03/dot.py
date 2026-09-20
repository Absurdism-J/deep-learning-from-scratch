import numpy as np
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[7,8],[9,0]])
C = np.array([2,3])
D = np.array([4,5,6])
if A.shape[1] == B.shape[0]:
    # print(np.dot(A,B))
    print(A.dot(B))
if A.shape[1] == C.shape[0]:
    print(A.dot(C))
if A.shape[0] == D.shape[0]:
    print(D.dot(A))