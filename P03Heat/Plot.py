
# coding: utf-8

# In[1]:


import shelve
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
from matplotlib import pyplot


# In[2]:


with shelve.open('Result') as db:
    n = db["1n"]
    m_list = db["1m"]
    rt = db["1rt"]


# In[3]:


print("HL")
print("$ 1 / \\tau $ CR $\\mu$ CR Explicit CR Implicit CR Crank--Nicolson BL")
print("HL")
for i, m in enumerate(m_list):
    print("{} CR {:.2e} CR {:.5e} CR {:.5e} CR {:.5e} BL".format(
        m,
        n**2 / m,
        rt[0][i],
        rt[1][i],
        rt[2][i],
    ))
    print("HL")


# In[4]:


with shelve.open('Result') as db:
    rt = db["2rt"]


# In[5]:


mtds = [
    "SQRT",
    "GS",
    "SD",
    "CG",
    "MG",
]


# In[6]:


print("HL")
print("Method CR Time (\\Si{s}) CR Iter. CR Err. $u$ CR Rel. $u$ CR Err. $u_{\\text{s}}$ CR Rel. $u_{\\text{s}}$ BL")
print("HL")
for i in range(5):
    print("{} CR {:.5f} CR {} CR {:.5e} CR {:.3e} CR {:.5e} CR {:.3e} BL".format(mtds[i], *rt[0][i]))
    print("HL")


# In[7]:


print("HL")
print("Method CR Time (\\Si{s}) CR Iter. CR Err. $u$ CR Rel. $u$ CR Err. $u_{\\text{s}}$ CR Rel. $u_{\\text{s}}$ BL")
print("HL")
for i in range(5):
    print("{} CR {:.5f} CR {} CR {:.5e} CR {:.3e} CR {:.5e} CR {:.3e} BL".format(mtds[i], *rt[1][i]))
    print("HL")


# In[8]:


with shelve.open('Result') as db:
    m_list = db["31m"]
    rt = db["31rt"]


# In[9]:


pyplot.figure(figsize=(6.0, 4.5))
pyplot.scatter(m_list[14:], rt[0][14:], 5.0)
pyplot.plot(m_list[14:], rt[0][14:], label="Exp.")
pyplot.scatter(m_list, rt[1], 5.0)
pyplot.plot(m_list, rt[1], label="Imp.")
pyplot.scatter(m_list, rt[2], 5.0)
pyplot.plot(m_list, rt[2], label="CN")
pyplot.semilogx()
pyplot.semilogy()
pyplot.ylim(1.0e-12, 3.0e-7)
pyplot.xlabel("$ 1 / \\tau $")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()
pyplot.savefig("Figure01.pgf")
pyplot.close()


# In[10]:


with shelve.open('Result') as db:
    n_list = db["32n"]
    rt = db["32rt"]


# In[11]:


labels = [
    "Exp., $ \\tau = h^2 / 4 $",
    "Exp., $ \\tau = h^2 / 8 $",
    "Exp., $ \\tau = h^2 / 12 $",
    "Exp., $ \\tau = h^2 / 16 $",
    "Imp., $ \\tau = h $",
    "Imp., $ \\tau = h / 4 $",
    "Imp., $ \\tau = h / 16 $",
    "CN, $ \\tau = h / 4 $",
    "CN, $ \\tau = h / 6 $",
    "CN, $ \\tau = h / 8 $",
]
styles = [
    "-",
    "-",
    "-",
    "-",
    "--",
    "--",
    "--",
    ":",
    ":",
    ":",
]
colors = [
    0.0,
    0.33,
    0.67,
    1.0,
    0.0,
    0.5,
    1.0,
    0.0,
    0.5,
    1.0,
]
cmap = pyplot.get_cmap("plasma")
cnorm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.2)
cmpbl = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)


# In[12]:


pyplot.figure(figsize=(6.0, 4.5))
for i in range(10):
    pyplot.scatter(n_list, rt[i], 5.0, c=cmpbl.to_rgba(colors[i]))
    pyplot.plot(n_list, rt[i], c=cmpbl.to_rgba(colors[i]), label=labels[i], linestyle=styles[i])
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$ 1 / h $")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()
pyplot.savefig("Figure02.pgf")
pyplot.close()

