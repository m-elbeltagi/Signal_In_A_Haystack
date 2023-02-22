## new 6 December 21, 2018 3:26 PM
from __future__ import division
import ROOT
import numpy as np
import math
signal_A = []
signal_B = []
signal_C = []
bkg_A = []
bkg_B = []
bkg_C = []
f = ROOT.TFile.Open("Mohamed_training_sets.root")
if f == ROOT.nullptr:
fatal("Could not open file")
tree_signal = f.Get("signal")
if tree_signal == ROOT.nullptr:
fatal("Failed accessing signal tree")
tree_bkg = f.Get("background")
if tree_bkg == ROOT.nullptr:
fatal("Failed accessing background tree")
nsignal_events = tree_signal.GetEntries()
nbkg_events = tree_bkg.GetEntries() #same as nsignal_events
for ievent in range(nsignal_events):
tree_signal.GetEntry(ievent)
tree_bkg.GetEntry(ievent)
#storing a,b,c in lists
signal_A.append(tree_signal.a)
signal_B.append(tree_signal.b)
signal_C.append(tree_signal.c)
bkg_A.append(tree_bkg.a)
bkg_B.append(tree_bkg.b)
bkg_C.append(tree_bkg.c)
###code used to draw the 1d and 2d histograms, same code was used replacing the variables
for each particular plot
#can1 = ROOT.TCanvas()
##hist1 = ROOT.TH1D("", "background_c", 100, -10, 5)
#hist1 = ROOT.TH2D("", "'background_b' vs 'background_c' scatter plot", 1000, -20, 20, 1000,
20, 20)
##binWidth = (hist1.GetXaxis().GetXmax() - hist1.GetXaxis().GetXmin())/hist1.GetNbinsX()
#
#for i in range(1000):
# hist1.Fill(bkg_C[i], bkg_B[i])
#hist1.GetXaxis().SetTitle('background_c')
#hist1.GetYaxis().SetTitle('background_b')
#hist1.SetMarkerStyle(20)
#hist1.SetMarkerSize(0.5)
###
#for num in bkg_C:
## hist1.Fill(num)
##hist1.GetXaxis().SetTitle('background_c')
##
#scale = 1/(hist1.Integral()*binWidth) #Integral() counts number of entries
##hist1.Scale(scale) #Normalized with respect to area
#hist1.Draw()
###changing lists to numpy arrays, to calculate covariance matrices:
signal_A = np.asarray(signal_A)
signal_B = np.asarray(signal_B)
signal_C = np.asarray(signal_C)
bkg_A = np.asarray(bkg_A)
bkg_B = np.asarray(bkg_B)
bkg_C = np.asarray(bkg_C)
signal_means = np.array([np.mean(signal_A), np.mean(signal_B), np.mean(signal_C)])
bkg_means = np.array([np.mean(bkg_A), np.mean(bkg_B), np.mean(bkg_C)])
sigVaa = np.cov(signal_A, signal_B)[0][0]
sigVbb = np.cov(signal_A, signal_B)[1][1]
sigVab = np.cov(signal_A, signal_B)[0][1]
sigVcc = np.cov(signal_A, signal_C)[1][1]
sigVac = np.cov(signal_A, signal_C)[0][1]
sigVbc = np.cov(signal_B, signal_C)[0][1]
#now we can define the covariance matrix for the signal
V_signal = np.array([[sigVaa, sigVab, sigVac], [sigVab, sigVbb, sigVbc], [sigVac, sigVbc,
sigVcc]])
# now repeating but for background:
bkgVaa = np.cov(bkg_A, bkg_B)[0][0]
bkgVbb = np.cov(bkg_A, bkg_B)[1][1]
bkgVab = np.cov(bkg_A, bkg_B)[0][1]
bkgVcc = np.cov(bkg_A, bkg_C)[1][1]
bkgVac = np.cov(bkg_A, bkg_C)[0][1]
bkgVbc = np.cov(bkg_B, bkg_C)[0][1]
V_bkg = np.array([[bkgVaa, bkgVab, bkgVac], [bkgVab, bkgVbb, bkgVbc], [bkgVac, bkgVbc,
bkgVcc]])
###the W matrix that will be used to define the fisher discriminant weights:
W = V_bkg + V_signal
Winv = np.linalg.inv(W)
###we can now find the test statistic t weights (so t = K_transpose . x):
K = np.dot(Winv, (bkg_means - signal_means))
###now we find the mean and std dev of t:
tsignal_mean = np.dot(K, signal_means)
tsignal_var = np.dot(K, np.dot(V_signal, K))
tsignal_stdev = tsignal_var**0.5
tbkg_mean = np.dot(K, bkg_means)
tbkg_var = np.dot(K, np.dot(V_bkg, K))
tbkg_stdev = tbkg_var**0.5
t_signal =[]
t_bkg = []
for i in range(1000):
t_signal.append(K[0]*signal_A[i] + K[1]*signal_B[i] + K[2]*signal_C[i])
t_bkg.append(K[0]*bkg_A[i] + K[1]*bkg_B[i] + K[2]*bkg_C[i])
t_all = t_signal + t_bkg
###we can now define our significance level, and use that to find tcut, so apply to mystery
data:
###note mean bkg (our null hypothesis H0 backgorund mean is larger than mean signal our H1):
alpha = 0.05
tcut = []
for j in t_bkg:
if abs(ROOT.Math.normal_cdf(((j-tbkg_mean)/tbkg_stdev)) - alpha) < 0.0005:
tcut.append(j)
#this gives only one element, the closest to 0.05 cutoff
tcut = tcut[0]
###calculating efficiencies:
bkg_eff = ROOT.Math.normal_cdf((tcut - tbkg_mean)/tbkg_stdev) #very close to 0.05
beta = 1 - ROOT.Math.normal_cdf((tcut - tsignal_mean)/tsignal_stdev)
signal_eff = 1 - beta
###missclassification ratios, these are about the same as the efficiencies:
bkg_miss = []
signal_miss = []
for i in t_bkg:
if i < tcut:
bkg_miss.append(i)
for i in t_signal:
if i > tcut:
signal_miss.append(i)
bkg_ratio = len(bkg_miss)/len(t_bkg)
signal_ratio = len(signal_miss)/len(t_signal)
###I now have a value for tcut, below which any events are considered signal, we can apply
this to the mystery data
mystery_A = []
mystery_B = []
mystery_C = []
m = ROOT.TFile.Open("Mohamed_mystery_data.root")
if m == ROOT.nullptr:
fatal("Could not open file")
tree_mystery = m.Get("data")
if tree_mystery == ROOT.nullptr:
fatal("Failed accessing mystery tree")
n_mystery = tree_mystery.GetEntries()
for i in range(n_mystery):
tree_mystery.GetEntry(i)
mystery_A.append(tree_mystery.a)
mystery_B.append(tree_mystery.b)
mystery_C.append(tree_mystery.c)

t_mystery = []
signal_mystery = []
for i in range(1000):
t_mystery.append(K[0]*mystery_A[i] + K[1]*mystery_B[i] + K[2]*mystery_C[i])
#can1 = ROOT.TCanvas()
#hist1 = ROOT.TH1D("", "Mystery t-values histogram for signal and background", 100, -16, 0)
##binWidth = (hist1.GetXaxis().GetXmax() - hist1.GetXaxis().GetXmin())/hist1.GetNbinsX()
#
#for num in t_mystery:
# hist1.Fill(num)
#
#hist1.GetXaxis().SetTitle("t_values")
##scale = 1/(hist1.Integral()*binWidth) #Integral() counts number of entries
##hist1.Scale(scale) #Normalized with respect to area
#hist1.Draw()
##
line = ROOT.TLine(-7.8126, 0, -7.8126, 73)
#line.Draw('SAME')
for i in t_mystery:
if i < tcut:
signal_mystery.append(i)
print "number of signal events is " , len(signal_mystery)," with around
",round(beta*(len(signal_mystery)/signal_eff)), " signal events missclassified as background"
