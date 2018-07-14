
# coding: utf-8

# In[1]:


import shelve
import utils
import algos


# In[2]:


n_list = [
    8, 12, 16, 24,
    32, 48, 64, 96,
    128, 192, 256, 384,
    512,
]
s = 1024
t = 1.0


# In[3]:


rt = [[], [], [], [], [], [], [], [], [], []]


# In[4]:


for n in n_list:
    m = 4 * n**2
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[0].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done explicit 4n^2")
    m = 8 * n**2
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[1].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done explicit 8n^2")
    m = 12 * n**2
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[2].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done explicit 12n^2")
    m = 16 * n**2
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_exp(n, m, u, t)
    rt[3].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done explicit 16n^2")
    m = n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_imp(n, m, u, t)
    rt[4].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done implicit n")
    m = 4 * n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_imp(n, m, u, t)
    rt[5].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done implicit 4n")
    m = 16 * n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_imp(n, m, u, t)
    rt[6].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done implicit 16n")
    m = 4 * n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_cn(n, m, u, t)
    rt[7].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done CN 4n")
    m = 6 * n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_cn(n, m, u, t)
    rt[8].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done CN 6n")
    m = 8 * n
    u = utils.ana_sol(n, 0.0)
    u = algos.pde_cn(n, m, u, t)
    rt[9].append(utils.calc_err(n, s // n, u, t)[0])
    print("Done CN 8n")
    print("Done n = {}".format(n))


# In[6]:


with shelve.open('Result') as db:
    db["32n"] = n_list
    db["32rt"] = rt

