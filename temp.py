import pinocchio as pin, numpy as np
m = pin.buildModelFromUrdf(r"C:\Users\ceti\Documents\01_projects\teleop-simulator\models\urdf\franka_fr3\fr3_franka_hand.urdf")
M0 = pin.crba(m, m.createData(), pin.neutral(m)).copy()
m.armature = np.full(m.nv, 0.5)
M1 = pin.crba(m, m.createData(), pin.neutral(m)).copy()
print(np.diag(M1 - M0))   # 0.5 everywhere => armature applied; zeros => it isn't