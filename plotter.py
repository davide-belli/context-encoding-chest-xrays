#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math


# plot losses on a unique figure 'plot.png'
def plotter(D_G_zs, D_xs, Advs, L2s, G_tots, D_tots, points_per_epoch, name=""):
    x = list(range(len(D_tots)))
    log_4 = [-math.log(4)] * len(D_tots)
    D_gain = [-k for k in D_tots]  # Discriminator gain defined as negative cross-entropy
    
    vline_position = [points_per_epoch * (x + 1) for x in range(int(math.floor(len(D_tots) / points_per_epoch)))]
    plt.clf()
    plt.plot(x, D_G_zs, "g-", linewidth=0.5, label="p D(G(z))")
    plt.plot(x, D_xs, "r-", linewidth=0.5, label="p D(x)")
    plt.plot(x, D_gain, "b-", linewidth=0.5, label="Disciminator")
    plt.plot(x, log_4, "k--", linewidth=0.5, label="-log(4)")
    plt.xlabel('x200 iterations')
    plt.ylabel('value')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/main4"+name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, D_G_zs, "g-", linewidth=0.5, label="p D(G(z))")
    plt.plot(x, D_xs, "r-", linewidth=0.5, label="p D(x)")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/fake-real_probs"+name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, Advs, "b-", linewidth=0.5, label="Adversarial loss")
    plt.plot(x, L2s, "g-", linewidth=0.5, label="L2 loss")
    plt.plot(x, G_tots, "k-", linewidth=0.5, label="Tot Generator loss")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/gen_losses"+name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, D_tots, "b-", linewidth=0.5, label="Tot Discriminator loss")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/disc_losses"+name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return
