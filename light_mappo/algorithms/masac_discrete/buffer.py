import numpy as np

class ReplayBufferDMASAC:
    """
    存每步转移（含掩码）：
      s, obs[A], a_idx[A], r_team(标量), s', obs'[A], done, avail[A,NA], avail'[A,NA]
    """
    def __init__(self, capacity, A, obs_dim, state_dim, n_actions):
        self.capacity = int(capacity)
        self.A, self.Do, self.S, self.NA = A, obs_dim, state_dim, n_actions
        self.reset_storage()

    def reset_storage(self):
        C,A,Do,S,NA = self.capacity, self.A, self.Do, self.S, self.NA
        self.s     = np.zeros((C, S), dtype=np.float32)
        self.obs   = np.zeros((C, A, Do), dtype=np.float32)
        self.a     = np.zeros((C, A),    dtype=np.int64)
        self.r     = np.zeros((C, 1),    dtype=np.float32)
        self.s2    = np.zeros((C, S),    dtype=np.float32)
        self.obs2  = np.zeros((C, A, Do),dtype=np.float32)
        self.done  = np.zeros((C, 1),    dtype=np.float32)
        self.avl   = np.ones ((C, A, NA),dtype=np.float32)
        self.avl2  = np.ones ((C, A, NA),dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, s, obs, a_idx, r_team, s2, obs2, done, avail, avail2):
        i = self.ptr
        self.s[i]    = s
        self.obs[i]  = obs
        self.a[i]    = a_idx
        self.r[i,0]  = float(r_team)
        self.s2[i]   = s2
        self.obs2[i] = obs2
        self.done[i,0]= float(done)
        self.avl[i]  = avail
        self.avl2[i] = avail2
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=int(batch_size))
        return (self.s[idxs], self.obs[idxs], self.a[idxs], self.r[idxs],
                self.s2[idxs], self.obs2[idxs], self.done[idxs],
                self.avl[idxs], self.avl2[idxs])
