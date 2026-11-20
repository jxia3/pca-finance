"""Generate minimal report diagrams: python -m scripts.framework_figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

OUTPUT = Path('report/figures')
BLUE, TEAL, ORANGE = '#376b8b', '#438579', '#b66b44'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def components():
    fig, ax = plt.subplots(figsize=(10, 2.1))
    ax.set(xlim=(-.15, 11.15), ylim=(-.1, 2.1))
    ax.axis('off')
    for x, title, detail in [
        (0, 'Returns', 'History through today'),
        (3, 'Models', 'PCA, regression,\ncustom signals'),
        (6, 'Strategy', 'Choose target weights'),
        (9, 'Simulator', 'Update positions,\nvalue and fees')]:
        ax.add_patch(Rectangle((x, 1.25), 2, .6, facecolor='#f3f6f8',
                               edgecolor='#bbc6ce', linewidth=.8))
        ax.text(x+1, 1.55, title, ha='center', va='center', fontsize=11)
        ax.text(x+1, .94, detail, ha='center', va='top', fontsize=9, color='#52616a')
        if x < 9:
            ax.annotate('', xy=(x+2.9, 1.55), xytext=(x+2.1, 1.55),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax.plot([10,10,7], [.35,.12,.12], color='#9aa6af', lw=1)
    ax.annotate('', xy=(7,.5), xytext=(7,.12),
                arrowprops=dict(arrowstyle='->', color='#9aa6af', lw=1))
    ax.text(8.5,-.04,'Current weights',ha='center',va='top',fontsize=8.5,color='#52616a')
    save(fig,'framework_components')


def walkforward():
    fig, ax = plt.subplots(figsize=(9,4.3), layout='constrained')
    for i in range(6):
        y = 5-i
        close = 7+i
        last_pca = 7+(i//3)*3
        ax.broken_barh([(last_pca-7,7)],(y+.04,.30),facecolors=BLUE)
        ax.broken_barh([(close-3,3)],(y-.34,.30),facecolors=TEAL)
        ax.broken_barh([(close,1)],(y-.30,.60),facecolors=ORANGE)
        ax.plot([close,close],[y-.39,y+.39],color='#a4adb4',lw=1)
    ax.set(xlim=(-.1,13.2),ylim=(-.7,6),
           yticks=range(5,-1,-1),yticklabels=['Day t']+[f'Day t+{i}' for i in range(1,6)],
           xticks=[1,3,5,7,9,11,13],
           xticklabels=['t−6','t−4','t−2','t','t+2','t+4','t+6'])
    ax.tick_params(axis='both',length=0,labelsize=9,pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(color=BLUE,label='PCA fit window'),
                       Patch(color=TEAL,label='Regression fit window'),
                       Patch(color=ORANGE,label='Next return')],
              loc='upper left',bbox_to_anchor=(0,1.08),ncol=3,frameon=False,fontsize=9)
    ax.set_xlabel('Trading day',fontsize=9,labelpad=9,color='#52616a')
    save(fig,'framework_walkforward')


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    components()
    walkforward()
