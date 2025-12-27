function [t,y] = step_response(g,nsim) 

[yd,t] = step(g,nsim);
[ny,nu]=size(g);
aux = zeros(size(yd,1),1);
for i = 1:ny
    for j = 1:nu
        aux = aux + yd(:,i,j);
    end
    y(:,i) = aux;
    aux = zeros(size(yd,1),1);
end

% close all
% plot(t,y)
% legend('toggle')
