function [A,B,C,d0,Dd,F,Psi,N] = opom(Gz)

% Gz: matriz de fun��es de transfer�ncia discretas (tf)
% Assume sistema est�vel

[ny,nu] = size(Gz);

% n�mero m�ximo de polos por canal
m = zeros(ny,nu);
for i = 1:ny
    for j = 1:nu
        polos = pole(Gz(i,j));
        m(i,j) = length(polos);
    end
end
na = max(m(:));   % n�mero m�ximo de polos

% inicializa��es
d0 = zeros(ny,nu);

for i = 1:ny
    for j = 1:nu
        
        [num,den] = tfdata(Gz(i,j),'v');
        
        if all(num == 0)
            d = 0;
            r = 0;
        else
            [d,r] = residue(num,den);
        end
        
        % termo direto (ganho est�tico)
        d0(i,j) = dcgain(Gz(i,j));
        
        % res�duos din�micos
        ddaux = d(1:end-1);
        ddaux = [ddaux; zeros(na - length(ddaux),1)];
        dd(:,j) = ddaux;
        
        % polos discretos
        faux = r(1:end-1);
        faux = [faux; zeros(na - length(faux),1)];
        f(:,j) = faux;
    end
    
    Dd(:,:,i) = dd;
    F(:,:,i)  = f;
end

% empilhamento diagonal
Dd = diag(Dd(:));
F  = diag(F(:));

% matriz J
J = [];
for i = 1:nu
    aux = [zeros(na,i-1) ones(na,1) zeros(na,nu-i)];
    J = [J; aux];
end

% matrizes N e Psi
Psi = [];
N   = [];
phi = ones(1,nu*na);

for i = 1:ny
    N = [N; J];
    aux = [zeros(1,(i-1)*na*nu) phi zeros(1,(ny-i)*na*nu)];
    Psi = [Psi; aux];
end

nd = na*nu*ny;

% modelo OPOM
A = [ eye(ny)        zeros(ny,nd)
    zeros(nd,ny)  F ];

B = [ d0
    Dd*F*N ];

C = [ eye(ny)  Psi ];

end
