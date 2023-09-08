# Breaking into the vault of Loki

After years of careful investigation, you have reached the gate to Loki's vault in the icy mountains of Norway, where it is said that many great treasures and powerful weapons are hidden. The gate seems unbreakable, but you spot some ancient machinery with inscriptions in old runes. After some help from ChatGPT, you are able to translate the symbols and the whole message into modern English, and it reads:

If you can prove that the polynomial 

$$
\begin{aligned}
p(x) &= 69 +78x + 32x^2 + 65x^3 + 82x^4 + 71x^5 + 69x^6 + 78x^7 + 84x^8 + 73x^9 \newline &+78x^{10} + 65x^{11} + 32x^{12} + 78x^{13} + 65x^{14}+ 67x^{15} + 73x^{16} + 32x^{17} \newline
&+ 84x^{18} + 73x^{19} + 69x^{20} + 82x^{21} + 82x^{22} + 65 x^{23} 
\end{aligned}
$$

is equal to $3$ at $x = 1$ modulo the BLS12-381 $r$ parameter, then the gate will open.

Below is a long list of bytes representing the SRS that can be used to perform KZG commitments. The machinery, after careful examination, performs the KZG verification using pairings. There is only one open place where you can place a wooden tablet with your answer, comprising 48 bytes. You guess this should be the proof of the KZG scheme, providing the point in compressed form, following the ZCash standard. The other elements contain the commitment to $p(x)$, the desired value $3$, and the point $x=1$. You ask ChatGPT for enlightenment, but it suddenly collapses and only shows the message: fatal error. Is this just an impossible task? Perhaps there is some trick to get by Loki's challenge...