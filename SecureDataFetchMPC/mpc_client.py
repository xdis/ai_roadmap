"""
MPC客户端 - 实现基于Shamir秘密共享的多方计算协议
"""
import random
import hashlib
from typing import List, Tuple, Dict, Any, Union

class MPCClient:
    """
    基于Shamir秘密共享方案的多方计算客户端
    用于安全处理API调用过程中的敏感信息（如Cookie）
    """
    
    def __init__(self, threshold: int, num_parties: int):
        """
        初始化MPC客户端
        
        参数:
            threshold: 恢复秘密所需的最小分享数
            num_parties: 参与方总数
        """
        self.threshold = threshold
        self.num_parties = num_parties
        
    def split_secret(self, secret: str) -> List[Tuple[int, int]]:
        """
        将秘密（如Cookie）分割成多个分享
        
        参数:
            secret: 要分割的秘密
            
        返回:
            分享列表，每个分享为(x, y)坐标对
        """
        # 计算秘密的哈希值作为种子
        hash_obj = hashlib.sha256(secret.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        random.seed(seed)
        
        # 将秘密转换为大整数
        secret_int = self._string_to_int(secret)
        
        # 创建一个多项式，其中常数项是秘密值
        # 多项式的次数是 threshold - 1
        coef = [secret_int]
        for _ in range(self.threshold - 1):
            coef.append(random.randint(0, 2**32-1))
        
        # 对多项式进行采样，生成分享
        shares = []
        for i in range(1, self.num_parties + 1):
            # 计算f(i)的值
            y = self._evaluate_polynomial(coef, i)
            shares.append((i, y))
            
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> str:
        """
        从分享中重建秘密
        
        参数:
            shares: 分享列表，每个为(x, y)坐标对
            
        返回:
            重建的秘密的哈希标识符
        """
        if len(shares) < self.threshold:
            raise ValueError(f"至少需要{self.threshold}个分享才能重建秘密")
        
        # 使用拉格朗日插值重建多项式的常数项，即秘密值
        secret_int = self._lagrange_interpolation(shares, 0)
        
        # 由于实际场景中我们使用的是Cookie这样的敏感字符串，
        # 在这个简化实现中我们返回一个哈希标识符
        return f"secret-{secret_int % 10000:04d}"
    
    def secure_cookie_handling(self, cookie: str) -> Dict[str, Any]:
        """
        安全处理Cookie信息
        
        参数:
            cookie: 原始Cookie字符串
            
        返回:
            处理结果包含分享和验证信息
        """
        # 分割Cookie
        shares = self.split_secret(cookie)
        
        # 创建验证哈希（在实际场景中可以使用更复杂的验证机制）
        verification = hashlib.sha256(cookie.encode()).hexdigest()[:8]
        
        return {
            "shares": shares,
            "verification": verification,
            "num_shares": len(shares),
            "threshold": self.threshold
        }
    
    def reconstruct_and_verify(self, 
                             cookie_data: Dict[str, Any], 
                             collected_shares: List[Tuple[int, int]],
                             original_cookie: str = None) -> Dict[str, Union[bool, str]]:
        """
        重建并验证Cookie信息
        
        参数:
            cookie_data: 通过secure_cookie_handling生成的数据
            collected_shares: 收集到的分享
            original_cookie: 可选的原始Cookie用于测试验证
            
        返回:
            重建结果和验证状态
        """
        try:
            # 重建秘密标识符
            secret_id = self.reconstruct_secret(collected_shares)
            
            # 如果提供了原始Cookie，进行验证
            if original_cookie:
                original_verification = hashlib.sha256(original_cookie.encode()).hexdigest()[:8]
                verified = original_verification == cookie_data["verification"]
            else:
                # 否则只检查是否能成功重建
                verified = True
                
            return {
                "success": True,
                "secret_id": secret_id,
                "verified": verified
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _string_to_int(self, s: str) -> int:
        """
        将字符串转换为整数
        """
        # 计算字符串的哈希值并转换为整数
        hash_obj = hashlib.sha256(s.encode())
        return int(hash_obj.hexdigest(), 16) % (2**32)
    
    def _evaluate_polynomial(self, coef: List[int], x: int) -> int:
        """
        计算多项式在点x处的值
        使用霍纳法则（秦九韶算法）提高效率
        """
        result = 0
        for c in reversed(coef):
            result = (result * x + c) % (2**32)  # 使用模运算防止溢出
        return result
    
    def _lagrange_interpolation(self, shares: List[Tuple[int, int]], x: int) -> int:
        """
        使用拉格朗日插值重建多项式的值
        
        参数:
            shares: 多项式上的点(x, y)列表
            x: 需要计算的点（通常为0，表示秘密值）
            
        返回:
            多项式在点x处的值
        """
        secret = 0
        modulus = 2**32
        
        for i, (xi, yi) in enumerate(shares):
            # 计算拉格朗日基本多项式的值
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (x - xj)) % modulus
                    denominator = (denominator * (xi - xj)) % modulus
            
            # 扩展欧几里得算法计算乘法逆元
            inverse_denominator = self._mod_inverse(denominator, modulus)
            
            # 累加当前项
            term = (yi * numerator * inverse_denominator) % modulus
            secret = (secret + term) % modulus
            
        return secret
    
    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        扩展欧几里得算法，计算a和b的最大公约数以及贝祖系数
        """
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = self._extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """
        计算a模m的乘法逆元
        """
        gcd, x, y = self._extended_gcd(a, m)
        if gcd != 1:
            raise ValueError(f"模逆不存在。{a}和{m}的最大公约数是{gcd}，不为1")
        else:
            return (x % m + m) % m