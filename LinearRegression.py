import numpy as np

class LinearRegression:
    """class untuk menghitung regresi linear"""
    x_train: "np.ndarray | list | tuple"
    y_train: "np.ndarray | list | tuple"
    alpha: float
    weight: float
    error: float
    cost_function_type: str
    cost_function: function
    cost_derivative: function

    def __init__(
            self, 
            x_train = np.ndarray | list | tuple, 
            y_train = np.ndarray | list | tuple, 
            alpha = 0.001,
            weight = 1.0,
            error = 1.0
        ) -> None:
        """
        inisiasi awal class LinearRegression

        Parameters :
        ----------

        x_train:
        data training untuk model regresi linear

        y_train:
        data aktual dari datasets 

        alpha:
        learning rate dari model regresi linear

        weight:
        berat atau kemiringan dari model awal

        error:
        intercept atau error dari model awal
        """

        verificator = lambda variable : variable \
                if isinstance(variable, np.ndarray) \
                else np.array(variable)

        try:
            self.x_train = verificator(x_train)
            self.y_train = verificator(y_train)
        except TypeError as e:
            print("tipe data untuk x_train atau y_train salah")
            print(e)
        
        self.alpha = alpha
        self.weight = weight
        self.error = error
        self.cost_function = None

    def fit(
            self, 
            cost_function_type = "MAE"
        ) -> None:
        """
        fitting data ke model regresi linear

        Parameters :
        ----------

        cost_function:
        pilih cost function yang akan digunakan = MAE, RMSE, R-Squared, Huber Loss
        """
        match cost_function_type:
            case "MAE" :
                self.cost_function = MAE(weight=self.weight, error=self.error, alpha=self.alpha)
            case "RMSE" :
                self.cost_function = self.RMSE
                self.cost_derivative = self.RMSE_derivative
            case "R-Squared":
                self.cost_function = self.R_Squared
                self.cost_derivative = self.R_Squared_derivative
            case "Huber Loss":
                pass
            case _:
                print("tidak ada cost function dengan nama tersebut")
    
    def RMSE(self) -> np.ndarray:
        """hitung Root Mean Square Error dari model"""
        return np.sqrt( 
                (1/self.x_train.size) * np.square(
                        np.sum(
                            self.y_train - self.distanceFromPredictor()
                        )
                    )
            ) 
    
    def R_Squared(self) -> np.ndarray:
        """hitung R-Squared dari model"""
        return np.sum(
                    np.square(
                        self.y_train - self.distanceFromPredictor() 
                    )
            ) / np.sum(
                    np.square(
                        self.y_train - (1/self.x_train.size) * self.x_train
                    )
            )
    
    def Huber_Loss(self, delta) -> np.ndarray:
        """hitung Huber Loss dari Model"""
        if(np.abs(self.distanceFromPredictor()) <= delta):
            return (1/2) * np.square(self.distanceFromPredictor())
        else:
            return delta * (
                    np.abs(
                            self.distanceFromPredictor() 
                        ) - 1/2 * delta
                )

    def __str__(self):
        return str(LinearRegression.__annotations__)

class LossFunction:
    """template untuk loss function"""
    alpha: float
    weight: float
    error: float

    def __init__(self, weight=1.0, error=1.0, alpha=0.001) -> None:
        """inisiasi awal"""
        self.alpha = alpha
        self.weight = weight
        self.error = error

    def Prediction(self, x_train=np.ndarray) -> np.ndarray:
        """hasil prediksi dari model"""
        return (x_train * self.weight + self.error)

    def actualToPredictionDistance(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """perbedaan antara nilai aktual dan nilai prediksi"""
        return y_train - self.Prediction(x_train=x_train)

    def absOfActualToPrediction(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """nilai absolute dari jarak antara nilai aktual dan nilai prediksi"""
        return np.abs(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            )
        )

    def updateParameter(self,
        parameter=float,
        loss_derivative_result=float) -> tuple:
        """fungsi template untuk update parameter model"""
        return parameter - self.alpha * loss_derivative_result

    def returnParameter(self) -> tuple:
        """mengembalikan parameter ke model untuk proses prediksi"""
        return (self.weight, self.error)

    def __del__(self):
        self.weight = None
        self.error = None
        self.alpha = None
    
class MAE(LossFunction):
    """Class untuk loss function Mean Square Error (MAE)"""
    def __init__(self, weight=1, error=1, alpha=0.001):
        super().__init__(weight, error, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung Mean Absolute Error (MAE) dari model"""
        x_size = x_train.size
        return (
            1/x_size
            ) * np.sum(
                self.absOfActualToPrediction(
                    x_train=x_train,y_train=y_train
                )
            )

    def derivativeToWeight(
        self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function MAE respectively ke weight"""
        x_size = x_train.size
        return (-1 * x_train * 1/x_size) * np.sum(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.absOfActualToPrediction(
                x_train=x_train,y_train=y_train
            )
        )

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        x_size = x_train.size
        return (-1/x_size) * np.sum(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.absOfActualToPrediction(
                x_train=x_train, y_train=y_train
            )
        )

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight -= self.updateParameter(
            self, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error -= self.updateParameter(
            self.error, self.derivativeToError(x_train=x_train,y_train=y_train)
        )

class MSE(LossFunction):
    def __init__(self, weight=1, error=1, alpha=0.001):
        super().__init__(weight, error, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung Mean Absolute Error (MAE) dari model"""
        x_size = x_train.size
        return (
            1/x_size
            ) * np.sum(
                np.square(
                    self.actualToPredictionDistance(
                        x_train=x_train,y_train=y_train
                    )
                )
            )

    def derivativeToWeight(
        self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function MAE respectively ke weight"""
        x_size = x_train.size

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        x_size = x_train.size

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight -= self.updateParameter(
            self, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error -= self.updateParameter(
            self.error, self.derivativeToError(x_train=x_train,y_train=y_train)
        )

class RMSE(LossFunction):
    def __init__(self, weight=1, error=1, alpha=0.001):
        super().__init__(weight, error, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung Root Mean Square Error (RMSE) dari model"""
        x_size = x_train.size
        return np.sqrt(
            1/x_size
            ) * np.sum(
                    np.square(
                    self.actualToPredictionDistance(
                        x_train=x_train,y_train=y_train
                    )   
                )
            )

    def derivativeToWeight(
        self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function MAE respectively ke weight"""
        x_size = x_train.size

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        x_size = x_train.size

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight -= self.updateParameter(
            self, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error -= self.updateParameter(
            self.error, self.derivativeToError(x_train=x_train,y_train=y_train)
        )

data = LinearRegression((10, 20), np.arange(10))

print(data)