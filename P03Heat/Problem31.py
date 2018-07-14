
# coding: utf-8

# In[1]:


import shelve
import utils
import algos


# In[6]:


n = 32
s = 1024
m_list = [
    32, 48, 64, 96,
    128, 192, 256, 384,
    512, 768, 1024, 1536,
    2048, 3072, 4096, 6144,
    8192, 12288, 16384, 24576,
    32768, 49152, 65536, 98304,
    131072, 196608, 262144, 393216
]
t = 1.0


# In[7]:


rt = [[], [], []]


# In[8]:


for m in m_list:
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[0].append(utils.calc_err(n, s // n, u, t)[0])
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_imp(n, m, u, t)
    rt[1].append(utils.calc_err(n, s // n, u, t)[0])
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_cn(n, m, u, t)
    rt[2].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done m = {}".format(m))


# In[10]:


with shelve.open('Result') as db:
    db["31m"] = m_list
    db["31rt"] = rt

