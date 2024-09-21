sort :: (Ord a) => [a] -> [a]
sort []     = []
sort (x:xs)  = sort left ++ [x] ++ sort right
    where
        left = filter (<=x) xs
        right = [ a | a<-xs, a > x]