class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def compute_cost(X,Y,w,b):
        m = len(X)
        total_cost = 0
        for i in range(m):
            f_wb = (w * X[i]) + b
            cost = (f_wb - Y[i]) **2
            total_cost += cost
        total_cost = total_cost / (2 * m)
        return total_cost

    
    def compute_gradient(self,X,Y,w,b):
        m = len(X)
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = (w * X[i]) + b
            dj_dw_i = X[i] * (f_wb - Y[i])
            dj_db_i = f_wb - Y[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        print(dj_db, dj_dw)

        return dj_dw, dj_db
    
    def gradient_descent(self, X, Y, w, b,alpha, num_iter):
        m = len(X)
        for i in range(num_iter):
            dj_dw, dj_db = self.compute_gradient(X, Y, w, b)
            w = w - (alpha *dj_dw)
            b = b- (alpha * dj_db)

        self.w = w
        self.b = b

    def fit(self, X, Y):
        self.gradient_descent(X,Y, 0,0,0.01, 300)

    def predict(self, X):
        for i in range(len(X)):
            output = self.w * X[i] + self.b
            print(output)

        