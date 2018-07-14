
# coding: utf-8

# In[2]:


import time
import shelve
import utils
import algos


# In[3]:


n, m = 128, 512
t = 1.0
eps = 1.0e-6


# In[4]:


rt = [[], []]


# In[4]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_cho_py(n, m, u, t)
u_s = u
end = time.time()
rt[0].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[5]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_gs_py(n, m, u, t, eps, 50)
end = time.time()
rt[0].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[6]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_sd_py(n, m, u, t, eps)
end = time.time()
rt[0].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[7]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_cg_py(n, m, u, t, eps)
end = time.time()
rt[0].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[8]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_mg_py(n, m, u, t, [3, 3], eps , 1)
end = time.time()
rt[0].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[9]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_cho_c(n, m, u, t)
u_s = u
end = time.time()
rt[1].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[10]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_gs_c(n, m, u, t, eps, 50)
end = time.time()
rt[1].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[11]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_sd_c(n, m, u, t, eps)
end = time.time()
rt[1].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[12]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_cg_c(n, m, u, t, eps)
end = time.time()
rt[1].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[13]:


start = time.time()
u = utils.ana_sol(n, 0.0)
u, ctr = algos.pde_imp_mg_c(n, m, u, t, [3, 3], eps , 1)
end = time.time()
rt[1].append([end - start, ctr, *utils.calc_err_std(n, 2, u, t, u_s)])


# In[5]:


with shelve.open('Result') as db:
    db["2rt"] = rt

