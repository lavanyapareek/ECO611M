'''
Excersise Question : 

Given a street
            ___
|   m-1      m      m+1| 
|   m-2             m+2|
|   .               .  |
|   .               .  |
|   .               .  |
|   3               n-2|
|   2               n-1|
|   1               n  |

S_r = m(m-1)/2
S_l = n(n+1)/2 - m(m+1)/2 (Sum of n - m + 1 terms)
Condition : S_r = S_l
=> m(m-1) = n(n+1) - m(m-1)
=> 2*m^2 = n(n+1)
'''


'''
$$$$$$$ Approximation :

n^2 + n - 2m^2 = 0

solving for n : 
    n = (-1 + sqrt(1+8m^2))/2
    for a sufficently large m :
        sqrt(1+8m^2) = 2*sqrt(2)*m
        and -1 can be ignored when m is large.
        hence, n = floor(sqrt(2)*m)
        
    example :
        without approximation :  
            if m = 6
            n = (-1 + sqrt(1+ 8*36))/2
            n = (-1 + sqrt(289))/2
            n = 8

        with approximation : 
            if m = 6
            n = floor(sqrt(2)*m)
            n = floor(8.48528)
            n = 8

       $$$$$$$ Try to plot these functions. $$$$$$

       Google collab :(6, 8), (35, 49), (204, 288), (1189, 1681), (6930, 9800), (40391, 57121), (235416, 332928), (1372105, 1940449), (7997214, 11309768), (46611179, 65918161), (271669860, 384199200)]
       Java Online comppiler maxed out with Long.MAX_VALUE : 
        (6, 8)
        (35, 49)
        (204, 288)
        (1189, 1681)
        (6930, 9800)
        (40391, 57121)
        (235416, 332928)
        (1372105, 1940449)
        (7997214, 11309768)
        (46611179, 65918161)
        (271669860, 384199200)
        (1583407981, 2239277041)
'''
#Java Code :
'''
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static List<long[]> validPairs(long N) {
        List<long[]> res = new ArrayList<>();
        long left = 0;
        for (long n = 2; n < N; n++){
            left = left + 1;
            double m = Math.sqrt(n*(n+1)/2);
            if(m == (long) m){
                res.add(new long[]{(long) m, n});
            }
        }
        return res;
    }

    public static void main(String[] args) {
        long N = Long.MAX_VALUE;
        List<long[]> pairs = validPairs(N);
        for (long[] pair : pairs) {
            System.out.println("(" + pair[0] + ", " + pair[1] + ")");
        }
        System.out.println("Total pairs found: " + pairs.size());
    }
}

'''

def validPairs(N):  
    res = []
    for m in range(2, N):
        left = 2*m*m
        for n in range(m+1, N + 1):
            if n*(n+1) == left:
                res.append((m,n))
    return res
def validPairs2(N):
    res = []
    for m in range(2, N):
        left = 2*m*m
        n = int(left**0.5) #approximate value of n
        if n*(n+1) == left:
            res.append((m,n))
    return res
#pairs = validPairs(10000)
pairs = validPairs2(10000000)
print(pairs)

