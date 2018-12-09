import numpy
import math

class Non_stationary_BSP:
    D = []                                  # The real historical demand grouped by weekday, provided by the data, but only accessed after a day is over.
#     f = 1                                   # The fifo-lifo level, which should be 1 throughout this assignment.
    z = 2                                   # The safety factor, for this assignment it is set to 2.
    cw = 100                                # The cost over overstocking (waste) a product.
    cs = 500                                # The cost of being a product short.
    m = None                                # A fitted model that can be used for machine learning, it is necessary that the model has a .predict function.
    t = 1                                   # The current day starting at 1.
    S = 0                                   # The stock level, is always equal to mu + z*sigma.
    I_tot = 0                               # The total current inventory, sum over I.
    Q = []                                  # The order quantity for a certain day is equal to S-I_tot.
    mu = 0                                  # The predicted/expected demand for today. Can be set by moving average or some other function.
    sigma = 0                               # The predicted/expected standard deviation for today. Should be approximately sqrt(mu).
    mu_hist = []                            # The list of historical mu's grouped by weekday
    I = [0]*5                               # The inventory of the product, taking into account how long the product has been in the inventory
                                            # I[i] gives the inventory of product that has been in the inventory for i days.
    waste = []                              # The number of products wasted for each day.
    short = []                              # The number of products short for each day.
    avg_mu = 2.2211538461538463             # The average demand available in the data this is used to speed up the warming period.
    
    # The initialisation function leaves the BSP in the state it would be at the beginning of the day. This means that it requires the demand of that day as input.
    def __init__(self, D_in, m_in=None, z_in=2, cw_in=100, cs_in=500, t_in=1, Q_in=[], mu_hist_in = [], sigma_hist_in = [], I_in=[0]*5, waste_in = [], short_in = [], avg_mu_in = 2.2211538461538463):
        self.D = [D_in]
        self.m = m_in
#         self.f = f_in
        self.z = z_in
        self.cw = cw_in
        self.cs = cs_in
        self.t = t_in
        self.S = 0
        self.Q = Q_in
        self.mu_hist = mu_hist_in
        self.sigma_hist = sigma_hist_in
        self.I = I_in
        self.I_tot = sum(self.I)
        self.mu = 0
        self.sigma = 0
        self.waste = waste_in
        self.short = short_in
        self.avg_mu = avg_mu_in
    
    # At the beginning of the day, the order of yesterday is added to the inventory, 
    # each inventory item is shifted and the last items are thrown out.
    def update_inventory(self):
        self.waste.append(self.I[0])
        for i in range(len(self.I)-1):
            self.I[i] = self.I[i+1]
        if self.t == 1:
            self.I[-1] = 0
        else:
            self.I[-1] = self.Q[-1]
        self.I_tot = sum(self.I)

    # Given the demand of days t-14, t-13, t-7 and t-6, calculate the average moving weight prediction
    # for the today and tomorrow.
    def mv_avg_predict(self):
        if self.t==1:
            self.mu = 2*self.avg_mu
        elif self.t<15:
            self.mu = self.D[self.t-2]
        else:
            d_14, d_13, d_7, d_6 = self.D[self.t-15], self.D[self.t-14], self.D[self.t-8], self.D[self.t-7]
            self.mu = (d_14+d_13+d_7+d_6)/2
    
    # Using the fitted Random Forest Regressor model and an input in the shape (Temp_0, Rainfall_0, Temp_1, Rainfall_1) this predicts the demand for the next two days.
    def RFR_predict(self, X):
        self.mu = max(0, self.m.predict(numpy.array([X]))[0])
    
    # The sigma is calculated according to what I presume to be the correct formula
    def calc_sigma(self):
        if self.t < 3:
            self.sigma = 0
        else:
            D = numpy.array([self.D[i] + self.D[i+1] for i in range(len(self.D[:self.t-2]))]) - numpy.array(self.mu_hist[2:])
            self.sigma = numpy.sqrt(D.dot(D)/(self.t-2))
    
    def process_demand(self):
        demand = self.D[-1]
        if demand >= self.I_tot:
            self.I = [0]*5
            self.short.append(demand-self.I_tot)
        else:
            i = 1
            while demand>0:
                supply = self.I[-i]
                self.I[-1] = max(0, demand)
                demand = max(0, demand-supply)
                i+=1
            self.short.append(0)
            
    def progress_day(self, daily_demand, X_in=None):
        # At the beginning of the day, we append the mu-history with yesterday's mu and sigma.
        self.mu_hist.append(self.mu)
        self.sigma_hist.append(self.sigma)
        # Next we update the inventory with yesterday's order which should have come in today.
        self.update_inventory()
        # We then predict the new mu and sigma for today. If a model was supplied, we use it to predict the new mu, otherwise we use the moving average method.
        if self.m == None:
            self.mv_avg_predict()
        else:
            self.RFR_predict(X_in)
        self.calc_sigma()
        # The Stock level for tomorrow is determined, it is equal to mu + z*sigma
        self.S = round(self.mu + self.z*self.sigma)
        # The order quantity for tomorrow is equal to S minus the current inventory and the total order quantity is updated.
        self.Q.append(max(0, self.S-self.I_tot))
        # Next we let the customers buy stuff (as possible) and subtract the day's demand from our current inventory. Note that even though the model technically already has access to the demand of today, it is only used here AFTER we set Q.
        self.process_demand()
        # We increase the day counter to reflect today.
        self.t+=1
        # Finally we add the demand for the next day.
        self.D.append(daily_demand)
        
    def performance(self, wup=14):
        waste, short, Q = self.waste[wup:], self.short[wup:], self.Q[wup:]
        avg_cost = numpy.mean(waste)*self.cw + numpy.mean(short)*self.cs
        w_perc = numpy.sum(waste)/numpy.sum(Q)
        s_perc = numpy.sum(short)/numpy.sum(Q)
        print("Sum Q: ", numpy.sum(Q))
        print("sum waste: ", numpy.sum(waste))
        print("sum short: ", numpy.sum(short))
        return avg_cost, w_perc, s_perc
        