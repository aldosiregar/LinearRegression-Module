import numpy as np

class LinearRegression:
    """class untuk menghitung regresi linear"""
    x_train: "np.ndarray | list | tuple"
    y_train: "np.ndarray | list | tuple"
    alpha: float
    weight: float
    error: float
    iter = int
    loss_function_type: str
    loss_function: "LossFunction"

    def __init__(
            self, 
            x_train = np.ndarray | list | tuple, 
            y_train = np.ndarray | list | tuple, 
            alpha = 0.1,
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
        try:
            self.x_train = self.variableVerificator(x_train)
            self.y_train = self.variableVerificator(y_train)
        except TypeError as e:
            print("tipe data untuk x_train atau y_train salah")
            print(e)
        
        self.alpha = alpha
        self.weight = weight
        self.error = error
        self.cost_function = None

    def variableVerificator(self, variable):
        """verifikasi variable iterable atau sudah menjadi numpy"""
        if(isinstance(variable, np.ndarray)):
            return variable
        else:
            return np.array(variable).ravel()

    def fit(
            self, 
            loss_function_type = "MAE",
            iter=500
        ) -> None:
        """
        fitting data ke model regresi linear

        Parameters :
        ----------

        cost_function:
        pilih cost function yang akan digunakan = MAE, RMSE, R-Squared, Huber Loss
        """
        match loss_function_type:
            case "MAE" :
                self.loss_function = MAE(weight=self.weight, error=self.error, alpha=self.alpha)
            case "RMSE" :
                self.loss_function = RMSE(weight=self.weight, error=self.error, alpha=self.alpha)
            case "R-Squared":
                self.loss_function = RSquared(weight=self.weight, error=self.error, alpha=self.alpha)
            case "Huber Loss":
                pass
            case _:
                print("tidak ada loss function dengan nama tersebut")

        for _ in range(iter):
            self.loss_function.update(
                x_train=self.x_train, y_train=self.y_train
            )

        self.weight, self.error = self.loss_function.returnParameter()

    def predict(self, x_test=np.ndarray | list | tuple):
        try:
            x_test = self.variableVerificator(x_test).ravel()
        except TypeError as e:
            print("tipe data bukan tupple, list, atau np.ndarray")
            print(e)
        return x_test * self.weight + self.error

    def __str__(self):
        return f"weight : {self.weight}, error : {self.error}"

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
        loss_derivative_result=float) -> float:
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
        return (-1/x_size) * np.sum(
            x_train * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.absOfActualToPrediction(
                x_train=x_train,y_train=y_train
            )
        )

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function MAE respectively ke error"""
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
        self.weight = self.updateParameter(
            self.weight, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error = self.updateParameter(
            self.error, self.derivativeToError(x_train=x_train,y_train=y_train)
        )

class MSE(LossFunction):
    def __init__(self, weight=1, error=1, alpha=0.001):
        super().__init__(weight, error, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung Mean Square Error (MSE) dari model"""
        x_size = x_train.size
        return (1/x_size) * np.sum(
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
        """turunan dari loss function MSE respectively ke weight"""
        x_size = x_train.size
        return (-2/x_size) * np.sum(
            x_train * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            )
        )

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function MSE respectively ke error"""
        x_size = x_train.size
        return (-2/x_size) * np.sum(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            )
        )

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight = self.updateParameter(
            self.weight, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error = self.updateParameter(
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
            (1/x_size) * np.sum(
                np.square(
                    self.actualToPredictionDistance(
                        x_train=x_train,y_train=y_train
                    )   
                )
            )
        )

    def derivativeToWeight(
        self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function RMSE respectively ke weight"""
        x_size = x_train.size
        return (-1/x_size) * np.sum(
            x_train * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.loss(
                x_train=x_train, y_train=y_train
            )
        )

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function RMSE respectively ke weight"""
        x_size = x_train.size
        return (-1/x_size) * np.sum(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.loss(
                x_train=x_train, y_train=y_train
            )
        )

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight = self.updateParameter(
            self.weight, self.derivativeToWeight(
                x_train=x_train,y_train=y_train
            )
        )
        self.error = self.updateParameter(
            self.error, self.derivativeToError(
                x_train=x_train,y_train=y_train
            )
        )

class RSquared(LossFunction):
    def __init__(self, weight=1, error=1, alpha=0.001):
        super().__init__(weight, error, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung R-Squared (coefficient of determination) dari model"""
        y_mean = np.sum(y_train)/y_train.size
        return np.square(
            self.absOfActualToPrediction(
                x_train=x_train, y_train=y_train
            ) / np.square(
                y_train - y_mean
            )
        ) 

    def derivativeToWeight(
        self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function R-Squared respectively ke weight"""
        x_size = x_train.size
        y_mean = np.sum(y_train)/y_train.size
        return (-2/x_size) * np.sum(
            x_train * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / np.square(
                y_train - y_mean
            )
        )

    def derivativeToError(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """turunan dari loss function R-Squared respectively ke error"""
        x_size = x_train.size
        y_mean = np.sum(y_train)/y_train.size
        return (-2/x_size) * np.sum(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / np.square(
                y_train - y_mean
            )
        )

    def update(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> None:
        """update parameter yang ada pada model"""
        self.weight = self.updateParameter(
            self.weight, self.derivativeToWeight(x_train=x_train,y_train=y_train)
        )
        self.error = self.updateParameter(
            self.error, self.derivativeToError(x_train=x_train,y_train=y_train)
        )