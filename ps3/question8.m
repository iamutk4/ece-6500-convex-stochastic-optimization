% Define the objective function and its gradient
f = @(x) x(1)^2 + 9*x(2)^2;
grad_f = @(x) [2*x(1); 18*x(2)];

% Define the projection operator onto the feasible set
proj = @(x) max(x, 0);

% Define the constraint functions and their gradients
g1 = @(x) 2*x(1) + x(2) - 1;
g2 = @(x) x(1) + 3*x(2) - 1;
grad_g1 = @(x) [2; 1];
grad_g2 = @(x) [1; 3];

% Define the Lagrange multiplier function and its gradient
L = @(x,lambda) f(x) + lambda(1)*max(-g1(x), 0) + lambda(2)*max(-g2(x), 0);
grad_L = @(x,lambda) grad_f(x) + lambda(1)*grad_g1(x).*((g1(x) < 0)) + lambda(2)*grad_g2(x).*((g2(x) < 0));

% Set the initial guess, maximum number of iterations, and tolerance level
x0 = [1; 1];
maxIter = 1000;
tol = 1e-6;

% Set the step sizes to test
alphas = [0.01, 0.1, 0.5];

% Create a figure to show the contour plot and solution trajectory for each step size
figure;
for i = 1:length(alphas)
    % Initialize the variables for this step size
    alpha = alphas(i);
    x = x0;
    lambda = [0; 0];
    
    % Create a contour plot of the objective function and constraints
    subplot(length(alphas), 2, 2*i-1);
    x1_vals = linspace(-2, 2, 100);
    x2_vals = linspace(-2, 2, 100);
    [X1,X2] = meshgrid(x1_vals,x2_vals);
    Z = X1.^2 + 9*X2.^2;
    g1_vals = 2*X1 + X2 - 1;
    g2_vals = X1 + 3*X2 - 1;
    Z(g1_vals < 0 | g2_vals < 0) = NaN;
    contour(X1,X2,Z,50);
    hold on;
    xlabel('x1');
    ylabel('x2');
    title(['Step size = ', num2str(alpha)]);
    
    % Run the projection gradient method
    for iter = 1:maxIter
        % Compute the projected gradient
        proj_grad = proj(x - alpha*grad_L(x, lambda));
        
        % Update the Lagrange multipliers
        lambda = max(lambda - alpha*[max(-g1(x), 0); max(-g2(x), 0)], 0);
        
        % Update the solution
        x_old = x;
        x = proj(x - alpha*proj_grad);
        
        % Check for convergence
        if norm(x - x_old) < tol
            break;
        end
    end
    
    % Plot the solution trajectory on the contour plot
    plot(x(1), x(2), 'r*', 'MarkerSize', 10);
    xlabel('x1');
    ylabel('x2');
    title(['Step size = ', num2str(alpha), ', x = (', num2str(x(1)), ', ', num2str(x(2)), ')']);
end