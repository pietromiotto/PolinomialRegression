---
author:
- |
  Student: PIETRO MIOTTO\
  Student's email: miottpi@usi.ch
date: 22/10/2023
title: Assignment 1
---

# Polynomial regression {#polynomial-regression .unnumbered}

We are given the following problem, which aims to build and train a Deep
Learning Model for regressing a polynomial.

## The Problem {#the-problem .unnumbered}

Let $z \in \mathbb{R}$ and consider the polynomial
$$p(z) = \frac{z^4}{100} - z^3 + z^2 - 10z = \sum_{k=1}^{4} z^k w_k$$
where $w = [w_1, w_2, w_3, w_4]^T = [-10, 1, -1, \frac{1}{100}]^T$. This
polynomial can also be expressed as the dot product of two vectors,
namely $$p(z) = w^T x \quad x = [z, z^2, z^3, z^4]^T$$ Consider an
independent and identically distributed (i.i.d.) dataset
$D = \{(z_i, y_i)\}_{i=1}^N$, where $y_i = p(z_i) + \epsilon_i$, and
each $\epsilon_i$ is drawn from a normal distribution with mean zero and
standard deviation $\sigma$.

Now, assuming that the vector $w$ is unknown, linear regression could
estimate it using the dot-product form presented in Equation 2. To
achieve this, we can move to another dataset
$$D' := \{(x_i, y_i)\}_{i=1}^N \quad x_i = [z_i, z_i^2, z_i^3, z_i^4]^T$$
The task of this assignment is to perform polynomial regression using
gradient descent with PyTorch, even if a closed-form solution exists.

## Questions {#questions .unnumbered}

1.  Define a function `plot_polynomial(coeffs, z_range, color=’b’)`
    Where `coeffs` is a `np.array` containing
    $[w_0, w_1, w_2, w_3, w_4]^T$ ($w_0$ in this case is equal to 0),
    `z_range` is the interval $[z_{\text{min}}, z_{\text{max}}]$ of the
    $z$ variable; `color` represents a color. Use the function to plot
    the polynomial. Report and comment on the plot.

2.  Write a function
    `create_dataset(coeffs, z_range, sample_size, sigma, seed=42)` that
    generates the dataset $D'$. Here, `coeffs` is a `np.array`
    containing $[w_0, w_1, w_2, w_3, w_4]^T$, `z_range` is the interval
    $[z_{\text{min}}, z_{\text{max}}]$ of the $z$ variable;
    `sample_size` is the dimension of the sample; $\sigma$ is the
    standard deviation of the normal distribution from which
    $\epsilon_i$ are sampled; `seed` is the seed for the random
    procedure.

3.  Use the code of the previous point to generate data with the
    following parameters:

    -   Each $z_i$ should be in the interval $[-3, 3]$

    -   $\sigma = 0.5$

    -   Use a sample size of 500 for training data and a seed of 0

    -   Use a sample size of 500 for evaluation data and a seed of 1

4.  Define a function `visualize_data(X, y, coeffs, z_range, title="")`
    that plots the polynomial $p(z)$ and the generated data (train and
    evaluation), where $X, y$ are as returned from the function
    `create_dataset`, `coeffs` are the coefficients of the polynomial,
    `z_range` is the interval $[z_{\text{min}}, z_{\text{max}}]$ of the
    $z$ variable, and `title` may be helpful to distinguish between the
    training and the evaluation plots. Provide two plots containing both
    the true polynomial; in one, add a scatter plot with the training
    data, and in the other a scatter plot with the evaluation data. Use
    the function to visualize the data. Report and comment on the plots.

5.  Perform polynomial regression on $D$ using linear regression on $D'$
    and report some comments on the training procedure. Consider your
    training procedure good when the training loss is less than 0.5.
    This works in my code with a number of steps equal to 3000. In
    particular, explain:

    -   How you preprocessed the data.

    -   Which learning rate you used, and why. What happens if the
        learning rate is too small; what happens if the learning rate is
        too high, and why.

    -   If bias should be set as `True` or `False` in `torch.nn.Linear`
        and why.

    Use the data you generated at point 3.

6.  Plot the training and evaluation loss as functions of the iterations
    and report them in the same plot. If you use both steps and epochs,
    you can choose either of the two, as long as it is clear from the
    plot and the plot reports what we expect---namely, that the loss
    functions decrease.

7.  Plot the polynomial you obtained with your estimated coefficient as
    well as the original one in the same plot.

8.  Plot the value of the parameters at each iteration as well as the
    true value.

9.  Re-train the model with the following parameters:

    -   Each $z_i$ should be in the interval $[-3, 3]$

    -   $\sigma = 0.5$

    -   Use a sample size of 10 for training data and a seed of 0

    -   Use a sample size of 500 for evaluation data and a seed of 1

    -   Keep the learning rate you chose at the previous point.

    Report: A plot with both the training and evaluation loss as
    functions of the iterations in the same plot, and the polynomial you
    got with your estimated coefficient as well as the original one in
    the same plot. Comment on what is going on. Note: Do not overwrite
    the code; keep the original training and this new one as two
    separate parts.

# Report {#report .unnumbered}

1.  **Line of code: 15-37**\
    Here, I define a function called `plot_polynomial` that takes as an
    argument the weights (i.e., coefficients of the polynomial), a range
    `z_range` that defines the range of values onto which the function
    is plotted. Optional arguments are the color of the plotted
    function, the `toprint` value, which defines if the function must
    return or print the plot, and the `label` argument, where you can
    define a customized label for the plot. This function, using
    `np.linspace`, creates an array of 1000 real values between the
    given range (in this case $[-3,3]$). Each of the (1000) points of
    the polynomial can be computed as:
    $$p(z) = \boldsymbol{w}^T \boldsymbol{x}, \quad \boldsymbol{x} = \left[z, z^2, z^3, z^4\right]^T$$
    which is the dot product between the weight vector (transposed) and
    a vector containing all the powers of the given value up to the
    $4-$th grade. Here, I coded this operation as follows:

    ``` {.python language="Python"}
    y = np.array([np.dot(coeffs, 
                [i**0, i**1, i**2, i**3, i**4]) for i in x])
    ```

    In order to make the code more general-purpose, i.e., to avoid
    hard-coded variables, I decided to define a function `f`, which
    takes as inputs the $x$ values array and the `coeff`, and returns an
    array containing the points of the polynomial function, by iterating
    on the number of elements (and the elements) of the weight vector
    and computing the sum of each coefficient multiplied for the
    corresponding power.\
    \
    **Line of code:136**\
    Here I just call the `plot_polynomial` function with the
    coefficients of the given polynomial and `z_range` between $[-3:3]$
    The result is in image [1](#fig:PLOT1){reference-type="ref"
    reference="fig:PLOT1"}.

2.  **Line of code:39-45** Here I define the `create_dataset` function.
    This function takes as input `coefficients`, which are the weights
    of the polynomial, `z_range` as the range in which the x values are
    computed, `sample_size` which is the size of the dataset, i.e. how
    much datapoints to produce, and, lastly, `sigma` and `seed` which
    are used to perform the normal distribution from which
    $\varepsilon_i$ are sampled. Each $\varepsilon_i$ is used to
    introduce noise in the data. In this function $x$ is defined as
    `num_samples` datapoints onto the interval `z_range`, while $y$ is
    an array containing `num_samples` elements, where each element is
    the output of the polynomial for the given input + some noise. The
    function returns $x$ and $y$. Please note that $y$ is the actual
    dataset.

3.  **Line of Code: 121-145** We are in the main. At first I setup the
    device we are using. If the device is detected to be a MacOS, and
    the MacOs GPU is found, then the program will make use of that,
    otherwise it will use CPU for computation. If instead the device id
    detected to not be a MacOs, if there is a CUDA GPU it will make use
    of that, if no CUDA GPU compatibility is found, it will use CPU.
    Now, please note that I generated training data and evaluation data
    in two different ways. The first is to just call the
    `create_dataset` function with `seed=1` to get training data and
    with `seed=0` to get evaluation data. I did so just to satisfy
    exercise 4. For the rest of the code I will be using as training and
    evaluation data two different instances of the DataLoader Class.
    This will allow me to work with an iterable that provides batches of
    the original data. By iterating on batches of data instead of the
    original data, the model can perform better both in time and memory
    consumption. Please note that I performed Exercise 5 both using and
    not using batches of data. That is the reason why in the first part
    of exercise 5 I access data and labels as `trainloader.dataset.data`
    and `trainloader.dataset.target` (same for `eval_loader`). I will
    deepen the differences of using and not using the DataLoader at the
    last point of this document.

4.  **Line of code: 47-56**\
    Here I define the function `visualize_data` which plots both the
    polynomial (calling the function `plot_polynomial`) and the
    datapoints generated by the `create_dataset` function. The aim of
    this function is to visualize how our training and evaluation
    dataset is distributed among the polynomial and therefore to
    understand the characteristics of the data and its response to
    noise. In **Line of code: 155-156** I call this function with the
    values returned from create dataset. By Looking at both plots
    [2](#fig:PLOT2Training){reference-type="ref"
    reference="fig:PLOT2Training"} and
    [3](#fig:PLOT2Eval){reference-type="ref" reference="fig:PLOT2Eval"},
    we can observe a pretty similar spread and distribution of points
    among the polynomial. This means that the random seed chosen for
    evaluation and training datapoints doesn't have a noticeable impact,
    i.e. the polynomial is robust and represents well the underlying
    structure of data. This last observation is confirmed also by the
    fact that points are concentrated along the curve. This results are
    reassuring, since they show that the data creation was performed
    well and the polynomial function fits well data with a robust
    behaviour in regards of the noise.

5.  I answer this question in different points:

    -   **Data Pre-processing: 100-113** . The data preprocessing takes
        place in the Dataset Class. Here $x$ (the 500 values in range
        $[-3:3]$ and $y$ (the output value of the polynomial applied to
        each of these $x$ values) are returned by `createing_dataset`
        function. Since, as we will see, the model I define takes 5
        features as input (each feature is a power of the x value up to
        4), I need to transform the x values vector into a $5x500$
        matrix. At first, with:

                        np.array([x**i for i in range(5)])

        I am creating a $sample\_size$ x $5$ matrix. All is transformed
        into a tensor, which is transposed and saved in the device. I
        transposed it in order to obtain a matrix that has as rows the
        features and as columns the datapoints. The $y$ values are
        simply reshaped in order to get a column vector of
        $sample\_size$ length. All the resulting tensors are of
        ` datatype=float32` in order to get precise results with
        relatively low memory consumption. Also, gradient cannot be
        performed on integer tenors.The creation of data and labels
        takes place in the class constructor `__init__`. The constructor
        initializes the object attributes, which in this case are `data`
        and `target`. The `__len__` method returns the len of the input
        data. Lastly, the `__getitem__`method returns, for a given
        $idx$, the sample and the target of a given element. I am not
        going to make use of these two methods, but they must be defined
        since the `MyDataSet` class extends the parent class
        `torch.utils.data.Dataset`.

    -   **Choosen Learning Rate: 163**. I use a learning rate of
        $0.001$, which represents a small update of the model parameters
        during the training process. I choose this learning rate because
        it was able to find the minimum of the loss function in a
        reasonable amount of steps ($\approx3000$). By selecting a
        smaller gradient, the model takes way more steps to converge to
        good results ($\approx30.000$). On the other hand, by choosing a
        higher learning rate, the update of the model parameters (or the
        step trough the minimum of the loss function) is too big, and
        thus it just bounces around the gradient without ever decreasing
        the loss function value to a suitable amount.

    -   **Bias Should be set True or False?** In `nn.Linear`, the bias
        should be set to `False` since the polynomial we are trying to
        estimate does not have any bias term, i.e., there is no term or
        constant that introduces an offset or a shift to the input data.

    -   **Model: 115-126** I defined the Model Class as follows (the
        example is taken from the lectures).

            # *** Model Class **
            class PolModel(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super(PolModel, self).__init__()

                self.fc = nn.Linear(input_dim, output_dim, bias=False)
                self.weight = self.fc.weight

                def forward(self, x):
                    return self.fc(x)

        In this part of the code I define a Class `PolModel`, which
        extends the `nn.Module` class. The first method `__init__` is a
        constructor that initializes istances of the PolModel class with
        `input_dim` as the input feature dimension for the model, and
        `output_dim` as the output dimension of what the model returns
        (in this case a 1-D array of weights). The variable `fc` defines
        the first layer of this model, where \"fc\" stands for \"fully
        connceted\". In this case this layer performs a linear
        regression of the inputs, and, as we shall see in the `forward`
        method, I did not use any activation function for this layer.
        Then we define `self.weights` as the weights returned by the
        `nn.Linear`, i.e. the parameters that at each iteration are
        updated by the model. The `def forward(self, x)` defines the
        forward step of the model that, in this case, consists of simply
        passing the input value to the `self.fc` and returning the
        output. This means that the forward step of our model, before
        the backpropagaion occurs, is to perform a linear regression on
        the input (and calculate the loss of the function as we shall
        see in the main).

    -   **Line of Code: 158-196** In the main firstly I initialise an
        instance of the `PolModel`, called `model` with $5$ input
        features and $1$ output (the estimated weight vector). Then I
        define my loss function, in this case I decided to use the Mean
        Squared Error to compute the loss. The main reason I choose to
        use this function is because it is the most used in regression
        tasks, and also it is capable of really fast computation. Then I
        define SGD as the model optimizer. The loss validation, loss
        training and parameters are just empty arrays useful to store
        the values returned by the loss function in training and
        evaluation phase, and the parameters of the model at any step.
        Therefore I make use of them to plot these values over time
        (where by time here I mean the steps). The training phase and
        the evaluation phase take place inside a for loop that iterates
        until the training loss is less than $0.5$, i.e. when the
        difference between the estimated weights output and the targets
        is minimal. At each iteration of this loop, the number of steps
        is increased. Inside this loop, I set the model in training
        phase, then I clear the gradients of the model parameters and
        then I compute the loss between the model's predictions and the
        actual target values. After that I append the training loss
        value and the weights to the storing arrays, and proceed to the
        backpropagation phase. Here, for each parameter of the model,
        the optimal \"adjustment\" is calculated and, in the following
        line(`optimizer.step()`, applied: the parameters are updated
        with the values that aims to reduce the loss (i.e. is the same
        as taking a step to the minimum of the function). After that, I
        perform the evaluation phase without tracking gradients, which
        saves memory and computational time (also, since in the
        evaluation phase there is no backpropagation, there is no need
        for that). I then compute the \"validation loss\" and append its
        value to the `loss_validation` array. Please note that every 100
        steps I print the loss value, the evaluation loss value and the
        weights.

6.  **Line of Code: 198-206** Here we are in the main, and I am passing
    to the `plot_loss` function, the array containing the training and
    evaluation loss at each step and, with `np.arange` I am creating an
    array with number of element equal to the number of total steps,
    i.e. is the array of the steps the model took to reach a training
    loss $<0,5$.\
    **Line of Code: 60-70** The `plot_loss` function, with `toprint`
    argument `=True` prints the plot having on the x-axis the number of
    steps and on the y-axis the value of the loss. If `toprint` argument
    `=False`, then this function returns the same plot (without
    printing). By looking at [4](#fig:LOSS1){reference-type="ref"
    reference="fig:LOSS1"} we can see that both the training and the
    evaluation loss decrease togheteher at the same rate. This means
    that the model is not overfitting: if so, the evaluation loss would
    increase while the training loss decreases. This is extremely
    important, since we can now have empirical evidence that the model
    is fitting well and correctly assuming the underlying structure of
    data. It is important to notice that to reach a training loss
    $<0,5$, precisely $\approx 0.499$, the model took $\approx 2700$
    steps (of course at each run of the model the value slighlty
    changes).

7.  **Line of Code: 209-217** Here, in the main, I call the function
    `plot_polynomial` for the original polynomial and the evaluated one.
    The evaluated one is the plot of the polynomial having as
    coefficients the estimated weights. As we can see in plot
    [5](#fig:POL2){reference-type="ref" reference="fig:POL2"}, the
    estimated weights produces a polynomial which is quite identical to
    the one we were trying to learn. Of course, with more steps we may
    obtain much closer results. This is another empirical proof that
    shows how the model is able to fit data coherently.

8.  **Line of Code: 74-96** The `plot_parameter` function takes 3
    arguments:

    -   `parameters`: an array containing the the values of each
        coefficient at each step.

    -   `steps:` an array containing as much elements as the steps

    -   `original parameters:` an array containing the coefficients of
        the original polynomial. This code produces a plot with 4
        subplots. Each of this subplots shows how a single parameter is
        updated during each step (I do not consider the parameter for
        $x^0$).

    By adding to the same subplot also the real value of the parameter,
    we can see if its update, at the end, converges with the real value
    of the parameter. As shown in [6](#fig:PARAM1){reference-type="ref"
    reference="fig:PARAM1"}, all the coefficients of the polynomial are
    estimated correctly. This means that at each step the gradient of
    the loss function is updated correctly. With more steps the
    estimated value would appear closer to the real value. The
    `plot_parameter` function is called at **Line of Code: 220**

9.  **Line of Code: 223-280** Here, I am creating a copy of the model
    training and evaluation. The code is the same as above except for:

    -   **train_loader:** the train loader here refers to a different
        istance of the dataset, where data and targets are built using
        $10$ as `num_samples`. This means that the input matrix is now
        $5$ features x $10$ samples only.

    -   **num_steps:** in this case I don't use a while loop, but I
        hard-coded `num_steps=27`. As you can see in
        [7](#fig:LOSS2){reference-type="ref" reference="fig:LOSS2"},
        both the training and the evaluation loss, instead of
        decreasing, rise exponentially after the 27-th step. This is
        because, with this learning rate, at each step the parameter
        updating follows an exponential behaviour. The learning rate is
        too big to find a minimum, and it just \"bounces around\". With
        a smaller stepsize, the loss function is able to find the
        minimum. This is firstly due to the shrinking of the training
        dataset: small datasets provide limited information about the
        underlying data distribution and present data with high
        variability, thus the model may struggle to capture the true
        patterns underlying data. Data points with greater variability
        can in fact lead to fluctuating gradients and unstable
        optimization process. If you look at
        [8](#fig:PARAM2){reference-type="ref" reference="fig:PARAM2"}
        where I show how the parameters behave over time (in this case I
        choose $500$ steps), it is clear how the update of parameters
        follows a \"bouncing pattern\", thus the stepsize is too big to
        find the minimum. By looking at
        [9](#fig:PLOT3){reference-type="ref" reference="fig:PLOT3"}, we
        can see a representation of the estimated polynomial in
        comparison to the original one. It passes through $0$ because of
        the coefficient for $x^0$, but then it behaves exponentially.

    <!-- -->

    -   **Line of Code: 288-359** I also tried to perform exercise 5 by
        iterating on batches. Since it wasn't requested I don't describe
        this in detail. The code is basically the same except that all
        the optimization and backpropagation are performed inside the
        for loop that iterates on data batches, while the set of the
        model in `train()` and `eval()` mode is done outside. Also, note
        that at each iteration of the batch loop, datapoints and labels
        of that batch are saved in the device. I kept the same learning
        rate and I choose $16$ as number of batches since it was the one
        that, with this learning rate, gave me the best result in terms
        of steps taken. Also, for most of the cases, a batch size
        between $[8;64]$ is suggested. Performing this linear regression
        task with the use of DataLoader is astonishingly more fast: to
        reach a training loss of $\approx 0.450$, the model took only
        $\approx 70$ steps. I also set `shuffle=True` in order to
        introduce randomness into the training process, which can help
        the model generalize better. Also, because the ordering of
        samples is not important for this task, choosing the `shuffle`
        option helps the model to not be biased by the order of data.
        `shuffle` is equal to `True` only for the `train_loader`, since
        we do not need randomization in the evaluation phase. Also, for
        efficiency purpose, i set the `batch_size` in the `eval_loader`
        aas equal to the size of the dataset. Here I attached the
        resulting plots for the training and evaluation loss
        [11](#fig:LOSS3){reference-type="ref" reference="fig:LOSS3"}, as
        well as the estimated and the original polynomial
        [10](#fig:PLOT4){reference-type="ref" reference="fig:PLOT4"}. As
        we see the `model3` performs almost as well as `model` ( the one
        not relying on DataLoader), but with a lot less steps.

    # Images {#images .unnumbered}

    ![Plot of the Polynomial Function on interval
    $[-3:3]$](images/images/fig1.jpg){#fig:PLOT1 width="75%"}

    ![Plot of generated training data and polynomial
    function](images/images/Visual1.png){#fig:PLOT2Training width="75%"}

    ![Plot of generated evaluation data and polynomial
    function](images/images/visual2.png){#fig:PLOT2Eval width="75%"}

    ![Plot of training loss and evaluation loss values at each step for
    training dataset with `num_samples=500` and evaluation dataset with
    `num_samples=500`](images/images/losses1.png){#fig:LOSS1
    width="75%"}

    ![Plot of original and estimated Polynomial Function on interval
    $[-3:3]$. For training dataset with `num_samples=500` and evaluation
    dataset with `num_samples=500`
    ](images/images/originalEstimated1.png){#fig:POL2 width="75%"}

    ![Plot of parameters update at each step and original parameter
    value for training dataset with `num_samples=500` and evaluation
    dataset with
    `num_samples=500`](images/images/param1.png){#fig:PARAM1
    width="75%"}

    ![Plot of training loss and evaluation loss values at each step for
    training dataset with `num_samples=10` and evaluation dataset with
    `num_samples=500`](images/images/Loss2.png){#fig:LOSS2 width="75%"}

    ![Plot of parameters update at each step and original parameter
    value for training dataset with `num_samples=10` and evaluation
    dataset with
    `num_samples=500`](images/images/bouncing.png){#fig:PARAM2
    width="75%"}

    ![Plot of original and estimated Polynomial Function on interval
    $[-3:3]$. For training dataset with `num_samples=10` and evaluation
    dataset with `num_samples=500`](images/images/PLOT3.png){#fig:PLOT3
    width="75%"}

    ![Plot of original and estimated Polynomial Function on interval
    $[-3:3]$. For polynomial regression performed with DataLoader,
    training dataset with `num_samples=500` and evaluation dataset with
    `num_samples=500` ](images/images/PLOT_final.png){#fig:PLOT4
    width="75%"}

    ![Plot of parameters update at each step and original parameter
    value for training dataset with `num_samples=500` and evaluation
    dataset with `num_samples=500`. Perfortmed with
    DataLoader](images/images/LOSS_final.png){#fig:LOSS3 width="75%"}
