import matplotlib.pyplot as plt, mpld3

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) 
pstr = mpld3.fig_to_html(fig)
print (pstr)
