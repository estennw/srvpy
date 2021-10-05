"""
This submodule contains sample curves, represented using bsplines.
"""

# External dependencies:
import numpy as np
from scipy.interpolate import BSpline, UnivariateSpline
from svgpathtools import svg2paths


def _bspline(p,d=3):
    n = p.shape[0]
    k = np.r_[np.zeros(d+1),(np.arange(1,n-d))/(n-d),np.ones(d+1)]
    b = BSpline(k, p, d)

    m = n-d
    t = np.linspace(0,1,2*m+1)
    al = np.cumsum(np.r_[0,np.abs(np.diff(b(t)))])
    phi = UnivariateSpline(al/al[-1],t,s=0)
    return lambda t: b(phi(t))


pawn = _bspline(np.array([0.000-0.000j, 1.664+0.000j, 4.993-0.000j, 5.673-0.000j, 5.880+0.590j, 5.817+1.146j, 5.012+1.524j, 5.016+2.180j, 5.614+2.338j, 5.585+3.020j, 2.458+5.732j, 2.354+6.569j, 2.678+6.934j, 2.666+7.355j, 0.446+9.484j, -0.012+16.201j, -0.117+17.048j, 3.438+16.990j, 3.216+18.557j, 0.246+18.709j, 0.117+19.750j, 2.163+20.767j, 2.654+23.924j, 1.771+26.105j, 0.020+27.416j, -2.127+28.015j, -4.275+27.416j, -6.026+26.105j, -6.909+23.924j, -6.418+20.767j, -4.371+19.750j, -4.500+18.709j, -7.470+18.557j, -7.692+16.990j, -4.137+17.048j, -4.243+16.201j, -4.700+9.484j, -6.920+7.355j, -6.932+6.934j, -6.609+6.569j, -6.712+5.732j, -9.840+3.020j, -9.868+2.338j, -9.271+2.180j, -9.266+1.524j, -10.071+1.146j, -10.134+0.590j, -9.928-0.000j, -9.247-0.000j, -5.919+0.000j, -4.254-0.000j, ]))
knight = _bspline(np.array([0.000-0.000j, 1.800+0.000j, 5.400-0.000j, 6.683-0.000j, 7.170+0.354j, 7.170+0.811j, 7.170+1.239j, 6.595+1.623j, 6.595+2.169j, 7.200+2.316j, 7.200+2.965j, 5.754+4.131j, 3.629+6.093j, 3.576+8.332j, 5.112+10.069j, 5.156+10.489j, 4.751+10.674j, 4.743+11.175j, 5.857+11.198j, 5.857+11.493j, 5.857+14.524j, 5.783+15.373j, 4.618+19.061j, 0.679+22.321j, 0.127+22.797j, -0.298+22.491j, -0.605+22.867j, -0.317+23.194j, -0.582+23.469j, -1.059+23.311j, -1.293+23.571j, -0.994+23.943j, -1.505+24.357j, -1.882+25.303j, -0.792+26.632j, 1.230+26.417j, 2.781+25.524j, 3.511+24.962j, 3.762+24.387j, 4.131+25.154j, 4.563+25.254j, 5.031+25.316j, 5.119+25.629j, 4.072+26.172j, 4.282+26.650j, 5.042+26.522j, 5.532+27.101j, 5.739+27.869j, 5.488+28.813j, 4.603+29.418j, 4.529+29.683j, 4.721+30.391j, 4.190+30.465j, 2.461+30.845j, -0.118+32.162j, -3.688+32.531j, -4.042+32.619j, -4.057+32.988j, -3.895+34.065j, -4.278+34.847j, -4.647+34.832j, -5.473+33.401j, -6.477+32.796j, -8.888+31.823j, -8.810+31.364j, -8.767+30.878j, -9.933+29.890j, -10.508+28.861j, -10.253+28.480j, -10.060+28.116j, -10.629+26.758j, -10.460+25.213j, -10.150+25.021j, -9.796+24.741j, -9.796+23.546j, -9.752+21.967j, -9.339+21.894j, -9.014+21.525j, -9.191+20.064j, -9.093+18.535j, -8.743+18.160j, -8.531+17.895j, -9.029+16.494j, -9.345+15.283j, -9.176+14.901j, -8.896+14.443j, -9.855+13.027j, -10.698+11.519j, -10.336+11.124j, -8.940+11.166j, -8.917+10.674j, -9.323+10.489j, -9.279+10.069j, -7.743+8.332j, -7.796+6.093j, -9.921+4.131j, -11.366+2.965j, -11.366+2.316j, -10.762+2.169j, -10.762+1.623j, -11.337+1.239j, -11.337+0.811j, -11.337+0.354j, -10.850+0.000j, -9.567+0.000j, -5.967+0.000j, -4.167+0.000j, ]))
bishop = _bspline(np.array([0.000-0.000j, -2.241+0.000j, -6.724-0.000j, -7.391-0.000j, -7.691+0.509j, -7.691+1.070j, -7.692+1.842j, -7.691+2.437j, -6.868+2.818j, -6.829+3.501j, -7.457+3.902j, -7.445+4.625j, -5.187+6.231j, -3.389+8.008j, -3.379+8.693j, -3.732+8.949j, -3.790+9.459j, -1.717+11.089j, -0.550+16.673j, -0.498+20.197j, -0.498+20.903j, -4.648+20.530j, -4.580+21.745j, -2.948+21.871j, -2.830+22.330j, -3.255+22.605j, -3.149+23.222j, -1.310+23.560j, -1.300+24.115j, -1.300+25.536j, -1.300+26.002j, -2.435+26.072j, -2.435+26.793j, -2.114+26.934j, -2.056+27.466j, -3.592+28.986j, -3.400+33.125j, -1.406+35.993j, 1.120+38.791j, 1.120+39.602j, -0.042+40.162j, 0.617+41.492j, 2.128+41.555j, 3.639+41.492j, 4.298+40.162j, 3.136+39.602j, 3.224+38.840j, 3.751+38.225j, 4.311+37.681j, 4.424+36.976j, 4.220+36.614j, 2.101+32.952j, 2.721+32.486j, 4.501+35.619j, 4.985+36.405j, 5.466+36.276j, 5.927+35.530j, 7.655+33.125j, 7.847+28.986j, 6.311+27.466j, 6.370+26.934j, 6.690+26.793j, 6.690+26.072j, 5.556+26.002j, 5.556+25.536j, 5.556+24.115j, 5.566+23.560j, 7.404+23.222j, 7.510+22.605j, 7.086+22.330j, 7.203+21.871j, 8.835+21.745j, 8.904+20.530j, 4.754+20.903j, 4.754+20.197j, 4.806+16.673j, 5.973+11.089j, 8.046+9.459j, 7.987+8.949j, 7.635+8.693j, 7.644+8.008j, 9.442+6.231j, 11.701+4.625j, 11.712+3.902j, 11.084+3.501j, 11.124+2.818j, 11.947+2.437j, 11.947+1.842j, 11.947+1.070j, 11.947+0.509j, 11.646-0.000j, 10.979-0.000j, 6.497-0.000j, 4.256-0.000j, ]))
rook = _bspline(np.array([0.000-0.000j, -1.946+0.000j, -5.838-0.000j, -6.557-0.000j, -6.887+0.397j, -6.887+0.830j, -6.887+1.688j, -6.887+2.038j, -6.101+2.221j, -6.004+2.959j, -6.586+3.056j, -6.674+3.551j, -5.847+5.945j, -5.163+7.622j, -4.902+8.321j, -3.624+8.494j, -3.624+8.875j, -4.750+8.884j, -4.799+9.350j, -4.444+9.893j, -3.359+11.755j, -1.995+17.274j, -1.837+23.784j, -1.837+24.288j, -2.905+24.319j, -2.905+24.651j, -2.923+25.414j, -3.921+25.371j, -4.284+25.367j, -4.225+25.876j, -4.316+28.192j, -4.703+29.548j, -4.697+29.921j, -4.381+30.158j, -3.994+30.170j, -2.518+30.160j, -1.871+30.170j, -1.873+29.133j, -1.873+28.845j, -1.597+28.845j, -1.099+28.845j, -0.818+28.849j, -0.809+29.123j, -0.778+30.101j, -0.221+30.199j, 2.116+30.180j, 4.454+30.199j, 5.011+30.101j, 5.042+29.123j, 5.050+28.849j, 5.331+28.845j, 5.829+28.845j, 6.106+28.845j, 6.105+29.133j, 6.104+30.170j, 6.750+30.160j, 8.226+30.170j, 8.614+30.158j, 8.929+29.921j, 8.936+29.548j, 8.549+28.192j, 8.457+25.876j, 8.516+25.367j, 8.154+25.371j, 7.155+25.414j, 7.138+24.651j, 7.138+24.319j, 6.070+24.288j, 6.070+23.784j, 6.228+17.274j, 7.592+11.755j, 8.676+9.893j, 9.032+9.350j, 8.983+8.884j, 7.857+8.875j, 7.857+8.494j, 9.134+8.321j, 9.396+7.622j, 10.080+5.945j, 10.907+3.551j, 10.819+3.056j, 10.236+2.959j, 10.334+2.221j, 11.120+2.038j, 11.120+1.688j, 11.120+0.830j, 11.120+0.397j, 10.790-0.000j, 10.070-0.000j, 6.178+0.000j, 4.233-0.000j, ]))
queen = _bspline(np.array([0.000-0.000j, 2.536+0.000j, 7.607-0.000j, 8.475-0.000j, 8.766+0.430j, 8.766+1.058j, 8.767+1.521j, 8.766+2.216j, 7.644+2.620j, 7.632+3.703j, 8.517+3.974j, 8.599+4.862j, 6.235+7.393j, 4.222+10.089j, 4.175+10.913j, 4.494+11.581j, 4.611+12.499j, 3.821+13.173j, 1.960+15.959j, 0.993+23.302j, 0.695+29.865j, 0.695+30.887j, 5.722+30.753j, 5.722+32.122j, 3.374+32.511j, 3.307+33.007j, 3.870+33.205j, 3.870+33.668j, 1.819+33.784j, 1.819+34.429j, 1.819+35.140j, 1.819+35.801j, 2.745+35.950j, 2.745+36.479j, 2.084+36.942j, 2.077+37.306j, 2.034+39.770j, 2.329+42.631j, 3.638+44.417j, 4.171+44.790j, 3.873+45.314j, 3.204+45.732j, 2.905+44.944j, 2.131+44.998j, 1.443+44.946j, 1.339+45.705j, 0.881+45.720j, 0.284+45.714j, 0.255+44.944j, -0.581+44.954j, -1.397+44.999j, -1.231+46.004j, -0.644+46.915j, -0.928+47.934j, -1.528+48.340j, -2.091+48.420j, -2.654+48.340j, -3.254+47.934j, -3.538+46.915j, -2.951+46.004j, -2.785+44.999j, -3.601+44.954j, -4.437+44.944j, -4.466+45.714j, -5.063+45.720j, -5.521+45.705j, -5.625+44.946j, -6.313+44.998j, -7.087+44.944j, -7.386+45.732j, -8.055+45.314j, -8.353+44.790j, -7.820+44.417j, -6.511+42.631j, -6.216+39.770j, -6.259+37.306j, -6.265+36.942j, -6.927+36.479j, -6.927+35.950j, -6.001+35.801j, -6.001+35.140j, -6.001+34.429j, -6.001+33.784j, -8.051+33.668j, -8.051+33.205j, -7.489+33.007j, -7.555+32.511j, -9.903+32.122j, -9.903+30.753j, -4.876+30.887j, -4.876+29.865j, -5.175+23.302j, -6.142+15.959j, -8.002+13.173j, -8.792+12.499j, -8.675+11.581j, -8.357+10.913j, -8.404+10.089j, -10.417+7.393j, -12.781+4.862j, -12.699+3.974j, -11.814+3.703j, -11.826+2.620j, -12.948+2.216j, -12.948+1.521j, -12.948+1.058j, -12.948+0.430j, -12.656-0.000j, -11.788-0.000j, -6.717-0.000j, -4.182-0.000j, ]))
king = _bspline(np.array([0.000-0.000j, 2.555+0.000j, 7.664-0.000j, 8.466-0.000j, 8.715+0.360j, 8.715+0.747j, 8.715+1.798j, 8.715+2.098j, 7.901+2.566j, 7.885+3.182j, 8.720+3.566j, 8.769+4.215j, 8.134+5.091j, 5.782+7.442j, 3.255+9.992j, 3.154+10.984j, 3.320+11.454j, 3.694+11.854j, 3.735+12.284j, 3.217+13.055j, 1.709+17.658j, 0.908+24.107j, 0.753+31.075j, 0.745+31.623j, 3.984+31.761j, 5.323+31.927j, 5.405+32.564j, 2.905+33.102j, 2.905+33.643j, 2.905+33.975j, 2.905+34.465j, 1.328+34.279j, 1.328+34.860j, 1.328+35.705j, 1.328+36.334j, 2.132+36.367j, 2.165+37.085j, 1.608+37.195j, 1.559+37.674j, 1.847+38.250j, 3.751+43.053j, 3.707+43.741j, 3.553+44.073j, 3.182+44.156j, 2.877+44.156j, 0.913+44.156j, 0.526+44.156j, 0.526+44.488j, 0.526+44.875j, 0.526+45.235j, 0.194+45.235j, -0.111+45.235j, -0.272+45.386j, -0.304+45.899j, 0.083+46.176j, -0.015+46.566j, -0.532+46.763j, -0.553+47.255j, -0.553+47.836j, -0.138+47.836j, 0.609+47.836j, 0.941+47.836j, 1.058+48.124j, 1.058+48.444j, 1.058+49.846j, 1.058+50.122j, 0.990+50.260j, 0.568+50.260j, -0.568+50.260j, -0.752+50.260j, -0.821+50.444j, -0.756+50.758j, -0.409+51.514j, -1.002+52.443j, -2.125+52.731j, -3.246+52.443j, -3.840+51.514j, -3.493+50.758j, -3.428+50.444j, -3.497+50.260j, -3.681+50.260j, -4.818+50.260j, -5.239+50.260j, -5.308+50.122j, -5.308+49.846j, -5.308+48.444j, -5.308+48.124j, -5.190+47.836j, -4.858+47.836j, -4.111+47.836j, -3.696+47.836j, -3.696+47.255j, -3.717+46.763j, -4.235+46.566j, -4.332+46.176j, -3.945+45.899j, -3.978+45.386j, -4.139+45.235j, -4.443+45.235j, -4.775+45.235j, -4.775+44.875j, -4.775+44.488j, -4.775+44.156j, -5.162+44.156j, -7.127+44.156j, -7.431+44.156j, -7.802+44.073j, -7.957+43.741j, -8.001+43.053j, -6.097+38.250j, -5.808+37.674j, -5.857+37.195j, -6.414+37.085j, -6.381+36.367j, -5.577+36.334j, -5.577+35.705j, -5.577+34.860j, -5.577+34.279j, -7.154+34.465j, -7.154+33.975j, -7.154+33.643j, -7.154+33.102j, -9.654+32.564j, -9.573+31.927j, -8.233+31.761j, -4.995+31.623j, -5.003+31.075j, -5.157+24.107j, -5.958+17.658j, -7.467+13.055j, -7.984+12.284j, -7.943+11.854j, -7.569+11.454j, -7.403+10.984j, -7.504+9.992j, -10.032+7.442j, -12.383+5.091j, -13.018+4.215j, -12.969+3.566j, -12.134+3.182j, -12.151+2.566j, -12.964+2.098j, -12.964+1.798j, -12.964+0.747j, -12.964+0.360j, -12.715+0.000j, -11.913+0.000j, -6.804+0.000j, -4.249+0.000j, ]))
