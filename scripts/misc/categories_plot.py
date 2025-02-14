import numpy as np
from matplotlib import pyplot as plt
import uproot3 as uproot


def central_63(data):
    """
    Return the minimum interveral containing 63% of the data

    """
    data = np.sort(data)
    n = len(data)
    return data[int(n * 0.5 - 0.315 * n)], data[int(n * 0.5 + 0.315 * n)]


data = uproot.open("~/Documents/Work/mvaOpt/root/output.root")[
    "diphotonDumper/trees/ggh_125_13TeV_All;3"
]
print(data.keys())

hist, bins = np.histogram(data["DiphotonMVA"].array(), bins=100, range=(0, 1))
plt.plot([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)], hist)

boundaries = [0.7271, 0.8519, 0.9055, 0.9544, 1]
dipho_mask3 = (data["DiphotonMVA"].array() > boundaries[0]) & (
    data["DiphotonMVA"].array() < boundaries[1]
)
dipho_mask2 = (data["DiphotonMVA"].array() > boundaries[1]) & (
    data["DiphotonMVA"].array() < boundaries[2]
)
dipho_mask1 = (data["DiphotonMVA"].array() > boundaries[2]) & (
    data["DiphotonMVA"].array() < boundaries[3]
)
dipho_mask0 = (data["DiphotonMVA"].array() > boundaries[3]) & (
    data["DiphotonMVA"].array() < boundaries[4]
)
cat3 = data["CMS_hgg_mass"].array()[dipho_mask3]
cat2 = data["CMS_hgg_mass"].array()[dipho_mask2]
cat1 = data["CMS_hgg_mass"].array()[dipho_mask1]
cat0 = data["CMS_hgg_mass"].array()[dipho_mask0]
lumi = 59.74
weights3 = data["weight"].array()[dipho_mask3] * lumi
weights2 = data["weight"].array()[dipho_mask2] * lumi
weights1 = data["weight"].array()[dipho_mask1] * lumi
weights0 = data["weight"].array()[dipho_mask0] * lumi

all_cats = np.concatenate([cat0, cat1, cat2, cat3])
all_weights = np.concatenate([weights0, weights1, weights2, weights3])
low_cats = np.concatenate([cat0, cat1])
low_weights = np.concatenate([weights0, weights1])
high_cats = np.concatenate([cat2, cat3])
high_weights = np.concatenate([weights2, weights3])

cat3_hist, bins = np.histogram(cat3, bins=100, range=(105, 140), weights=weights3)
cat2_hist, bins = np.histogram(cat2, bins=100, range=(105, 140), weights=weights2)
cat1_hist, bins = np.histogram(cat1, bins=100, range=(105, 140), weights=weights1)
cat0_hist, bins = np.histogram(cat0, bins=100, range=(105, 140), weights=weights0)

all_cats_hist, bins = np.histogram(
    all_cats, bins=100, range=(105, 140), weights=all_weights
)
low_cats_hist, bins = np.histogram(
    low_cats, bins=100, range=(105, 140), weights=low_weights
)
high_cats_hist, bins = np.histogram(
    high_cats, bins=100, range=(105, 140), weights=high_weights
)


fig, ax = plt.subplots(3, 4, figsize=(20, 20))

mids = (bins[1:] + bins[:-1]) / 2

line1, line2 = central_63(all_cats)
events = all_weights[(all_cats > line1) & (all_cats < line2)]
ax[0, 0].plot(
    mids,
    all_cats_hist,
    label=f"1/1 category\nevents = {events.sum():.0f}\nmean = {all_cats.mean():.2f}\nstd = {all_cats.std():.2f}",
)
ax[0, 0].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[0, 0].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(low_cats)
events = low_weights[(low_cats > line1) & (low_cats < line2)]
ax[1, 0].plot(
    mids,
    low_cats_hist,
    label=f"1/2 categories\nevents = {events.sum():.0f}\nmean = {low_cats.mean():.2f}\nstd = {low_cats.std():.2f}",
)
ax[1, 0].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[1, 0].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(high_cats)
events = high_weights[(high_cats > line1) & (high_cats < line2)]
ax[1, 1].plot(
    mids,
    high_cats_hist,
    label=f"2/2 categories\nevents = {events.sum():.0f}\nmean = {high_cats.mean():.2f}\nstd = {high_cats.std():.2f}",
)
ax[1, 1].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[1, 1].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(cat0)
events = weights0[(cat0 > line1) & (cat0 < line2)]
ax[2, 0].plot(
    mids,
    cat0_hist,
    label=f"1/4 categories\nevents = {events.sum():.0f}\nmean = {cat0.mean():.2f}\nstd = {cat0.std():.2f}",
)
ax[2, 0].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[2, 0].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(cat1)
events = weights1[(cat1 > line1) & (cat1 < line2)]
ax[2, 1].plot(
    mids,
    cat1_hist,
    label=f"2/4 categories\nevents = {events.sum():.0f}\nmean = {cat1.mean():.2f}\nstd = {cat1.std():.2f}",
)
ax[2, 1].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[2, 1].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(cat2)
events = weights2[(cat2 > line1) & (cat2 < line2)]
ax[2, 2].plot(
    mids,
    cat2_hist,
    label=f"3/4 categories\nevents = {events.sum():.0f}\nmean = {cat2.mean():.2f}\nstd = {cat2.std():.2f}",
)
ax[2, 2].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[2, 2].axvline(line2, color="r", linestyle="dotted", linewidth=1)

line1, line2 = central_63(cat3)
events = weights3[(cat3 > line1) & (cat3 < line2)]
ax[2, 3].plot(
    mids,
    cat3_hist,
    label=f"4/4 categories\nevents = {events.sum():.0f}\nmean = {cat3.mean():.2f}\nstd = {cat3.std():.2f}",
)
ax[2, 3].axvline(line1, color="r", linestyle="dotted", linewidth=1)
ax[2, 3].axvline(line2, color="r", linestyle="dotted", linewidth=1)
# legend
ax[0, 0].legend()
ax[1, 0].legend()
ax[1, 1].legend()
ax[2, 0].legend()
ax[2, 1].legend()
ax[2, 2].legend()
ax[2, 3].legend()

# don't draw the empty plot
ax[0, 1].axis("off")
ax[0, 2].axis("off")
ax[0, 3].axis("off")
ax[1, 2].axis("off")
ax[1, 3].axis("off")


plt.show()

# add legend
