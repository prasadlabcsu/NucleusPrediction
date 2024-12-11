function [PER_EXP_H] = PER_EXP_VAR_W_SM(Height,M_Inter)

    PER_EXP_H = [];

    for k = 2:height(Height.Coefficients)
        Var = Height.CoefficientNames{k};
        if ~contains(Var,':')
            Var = str2double(Var(2:end));
            PER_EXP_H(length(PER_EXP_H)+1) = table2array(Height.Coefficients(k,1))*(1/(length(Height.Fitted) - 1))*sum(M_Inter(:,595 + Var).*Height.Fitted);
        else
            PER_EXP_H(length(PER_EXP_H)+1) = Height.Rsquared.Ordinary - sum(abs(PER_EXP_H));
            break
        end
    end
    
    PER_EXP_H = 100*Height.Rsquared.Ordinary*(abs(PER_EXP_H)/sum(abs(PER_EXP_H)));
    
end