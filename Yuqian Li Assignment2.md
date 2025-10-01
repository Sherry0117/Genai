# Yuqian Li — Assignment 2: Theory — Arithmetic of CNNs

---

## Q1
**Known**  
- Input: 32×32×3  
- Conv: 8 filters, kernel 5×5, stride s=1, padding p=0  

**Formula**  
$$ 
out = \left\lfloor \frac{in + 2p - k}{s} \right\rfloor + 1 
$$  

**Calculation**  
$$ 
out = \left\lfloor \frac{32 + 2\cdot0 - 5}{1} \right\rfloor + 1 
= \left\lfloor 27 \right\rfloor + 1 = 28 
$$  

Channels = 8  

**Answer**  
$$ 28 \times 28 \times 8 $$

---

## Q2
**Known**  
- Input: 32×32×3  
- Conv: 8 filters, kernel 5×5, stride s=1, padding = "same"  
- For odd kernel size, $$ p = \frac{k-1}{2} = 2 $$  

**Formula**  
$$ 
out = \left\lfloor \frac{32 + 2\cdot2 - 5}{1} \right\rfloor + 1 
$$  

**Calculation**  
$$ 
out = \left\lfloor \frac{31}{1} \right\rfloor + 1 = 32 
$$  

Channels = 8  

**Answer**  
$$ 32 \times 32 \times 8 $$

---

## Q3
**Known**  
- Input: 64×64  
- Conv: kernel 3×3, stride=2, padding=0  

**Formula**  
$$
out = \left\lfloor \frac{64 + 2\cdot0 - 3}{2} \right\rfloor + 1
$$  

**Calculation**  
$$
out = \left\lfloor \frac{61}{2} \right\rfloor + 1 = 30 + 1 = 31
$$  

**Answer**  
$$ 31 \times 31 $$  
(spatial size)


---

## Q4
**Known**  
- Input: 16×16×C  
- MaxPool: kernel 2×2, stride s=2, padding p=0  

**Formula**  
$$ 
out = \left\lfloor \frac{16 + 2\cdot0 - 2}{2} \right\rfloor + 1 
$$  

**Calculation**  
$$ 
out = \left\lfloor \frac{14}{2} \right\rfloor + 1 = 7 + 1 = 8 
$$  

Channels unchanged  

**Answer**  
$$ 8 \times 8 \times C $$

---

## Q5
**Known**  
- Input: 128×128  
- Two conv layers, each kernel 3×3, stride s=1, padding = "same" (p=1)  

**Formula per layer**  
$$ 
out = \left\lfloor \frac{128 + 2\cdot1 - 3}{1} \right\rfloor + 1 
$$  

**Calculation**  
$$ 
out = \left\lfloor 127 \right\rfloor + 1 = 128 
$$  

- After first conv: 128×128  
- After second conv: 128×128  

**Answer**  
$$ 128 \times 128 \times C_2 $$  
(where \(C_2\) = number of filters in the second conv)

---

## Q6
**Known**  
- PyTorch training requires calling `model.train()`  

**Explanation**  
- `model.train()` sets layers to training mode  
- **BatchNorm**: updates running mean/variance  
- **Dropout**: randomly zeroes activations for regularization  

**If omitted:**  
- BatchNorm will not update running stats  
- Dropout is disabled (acts like eval mode)  
- Training may degrade and evaluation mismatch occurs  

**Answer**  
Without `model.train()`, training is incorrect: dropout won’t apply, batch norm won’t update → degraded performance.
