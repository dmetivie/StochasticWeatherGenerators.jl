onefy(x::Number) = x == 0 ? 0 : 1
onefy(x::Missing) = missing
onefy(x::Number, eps) = abs(x) <= eps ? 0 : 1
onefy(x::Missing, eps) = missing
zerofy(x) = abs(x) < 1 ? 0.0 : x
zerofy(x, eps) = abs(x) < eps ? 0.0 : x
zerofy!(y, vmin) = map!(x -> abs(x) < vmin ? zero(x) : x, y)

vec_triu(A) = A[tril(trues(size(A)), -1)] # return striclty upper diagonal element of matrix A

# * Date * #

whole_year(y) = Dates.Date(y):Day(1):(Dates.Date(y + 1)-Day(1))

dayofyear_Leap(d) = @. dayofyear(d) + ((!isleapyear(d)) & (month(d) > 2))

# * Plots * #

"""
    my_color(k, K)
Convenience for plot colors and hidden states to blue for k=1 (∼wetter)  and orange for k=K (∼driest)
"""
function my_color(k, K)# 
    if k == 1
        return 1
    elseif k > 1 && k < K
        return k + 1
    elseif k == K
        return 2
    end
end
