import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('', 'Kopf','Zahl', '')
y_pos = np.arange(len(objects))
performance = [0, 0.5,0.5, 0]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Wahrscheinlichkeit des Ergebnisses')
plt.xlabel('Ergebnis')
plt.title('Diskrete Verteilung')
plt.tight_layout()

plt.show()