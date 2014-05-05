from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft
# Colors.
integrate_color = {"ec": "#bbbbbb"}
optimize_color = {"ec": "#f89406"}


# Instantiate the PGM.
pgm = daft.PGM([5.3, 3.05], origin=[0.3, 0.3])

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha_0", r"$\alpha_0$", 2, 3, fixed=True))
pgm.add_node(daft.Node("alpha_1", r"$\alpha_1$", 3, 3, fixed=True))
pgm.add_node(daft.Node("alpha_2", r"$\alpha_2$", 4, 3, fixed=True))
pgm.add_node(daft.Node("alpha_3", r"$\alpha_3$", 5, 3, fixed=True))
pgm.add_node(daft.Node("theta_prior", r"$\beta$", 1, 3, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("P_0", r"$\mathbf{p}_0$", 2, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("P_1", r"$\mathbf{p}_1$", 3, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("P_2", r"$\mathbf{p}_2$", 4, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("P_3", r"$\mathbf{p}_3$", 5, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("theta", r"$\theta$", 1, 2, plot_params=optimize_color))

# Data.
pgm.add_node(daft.Node("n_0", r"$n_0$", 2, 1, observed=True))
pgm.add_node(daft.Node("n_1", r"$n_1$", 3, 1, observed=True))
pgm.add_node(daft.Node("n_2", r"$n_2$", 4, 1, observed=True))
pgm.add_node(daft.Node("n_3", r"$n_3$", 5, 1, observed=True))

# Add in the edges.
pgm.add_edge("alpha_0", "P_0")
pgm.add_edge("alpha_1", "P_1")
pgm.add_edge("alpha_2", "P_2")
pgm.add_edge("alpha_3", "P_3")
pgm.add_edge("P_0", "n_0")
pgm.add_edge("P_1", "n_1")
pgm.add_edge("P_2", "n_2")
pgm.add_edge("P_3", "n_3")
pgm.add_edge("theta_prior", "theta")
pgm.add_edge("theta", "n_0")
pgm.add_edge("theta", "n_1")
pgm.add_edge("theta", "n_2")
pgm.add_edge("theta", "n_3")

# And a plate.
#pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("skynetPGMverbose.pdf")
pgm.figure.savefig("skynetPGMverbose.png", dpi=150)


#-----------------------------------------------------------------------------------------------------------------------
# Another one, supposedly equivalent
pgm = daft.PGM([5.3, 2.05], origin=[0.3, 0.3])

# Hierarchical parameters.
pgm.add_node(daft.Node("theta_prior", r"$\beta$", 5, 1, fixed=True))
pgm.add_node(daft.Node("alpha", r"$\alpha_k$", 1, 1, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("theta", r"$\theta$", 4, 1, plot_params=optimize_color))
pgm.add_node(daft.Node("p", r"$\mathbf{p}_k$", 2, 1, plot_params=integrate_color))

# Data.
pgm.add_node(daft.Node("n", r"$\mathbf{n}_k$", 3, 1, observed=True))

# Add in the edges.
pgm.add_edge("alpha", "p")
pgm.add_edge("p", "n")
pgm.add_edge("theta_prior", "theta")
pgm.add_edge("theta", "n")

# And a plate.
pgm.add_plate(daft.Plate([0.5, 0.5, 3, 1], label=r"$k = 1, \cdots, K$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("skynetPGM.pdf")
pgm.figure.savefig("skynetPGM.png", dpi=150)
