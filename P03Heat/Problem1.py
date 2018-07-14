
# coding: utf-8

# In[1]:


import shelve
import numpy
import utils
import algos


# In[2]:


n = 32
m_list = [32, 128, 512, 2048, 3072, 3584, 3840, 4096, 8192, 16384, 32768]
t = 1.0


# In[3]:


rt = [[], [], []]


# In[4]:


for m in m_list:
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[0].append(numpy.linalg.norm(u, numpy.infty))
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_imp(n, m, u, t)
    rt[1].append(numpy.linalg.norm(u, numpy.infty))
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_cn(n, m, u, t)
    rt[2].append(numpy.linalg.norm(u, numpy.infty))
    print("Done m = {}".format(m))


# In[5]:


with shelve.open('Result') as db:
    db["1n"] = n
    db["1m"] = m_list
    db["1rt"] = rt

