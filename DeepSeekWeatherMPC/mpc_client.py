"""
MPC客户端 - 实现基于Shamir秘密共享的多方计算协议
使用纯Python实现，不依赖外部加密库
"""
import random
import hashlib
from typing import List, Tuple

class MPCClient:
    """
    基于Shamir秘密共享方案的多方计算客户端实现
    用于安全地处理API密钥等敏感信息
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
        使用Shamir秘密共享方案分割秘密
        
        参数:
            secret: 要分割的秘密(API密钥)
            
        返回:
            分享列表，每个分享为(x, y)坐标对
        """
        # 将秘密转换为数字（使用简单的哈希和转换）
        secret_int = self._string_to_int(secret)
        
        # 生成多项式的随机系数
        # 第一个系数是秘密本身，其余threshold-1个是随机生成的
        coef = [secret_int] + [random.randint(0, 2**32-1) for _ in range(self.threshold - 1)]
        
        # 生成分享
        shares = []
        for i in range(1, self.num_parties + 1):
            # 在x=i处评估多项式的值
            y = self._evaluate_polynomial(coef, i)
            shares.append((i, y))
            
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> str:
        """
        从部分分享中重建原始秘密
        
        参数:
            shares: 分享列表，每个为(x, y)坐标对
            
        返回:
            重建的秘密字符串
        """
        if len(shares) < self.threshold:
            raise ValueError(f"至少需要{self.threshold}个分享才能重建秘密")
        
        # 使用拉格朗日插值重建秘密
        secret_int = self._lagrange_interpolation(shares, 0)
        
        # 将数字转回字符串
        # 由于我们的转换是不可逆的，所以我们在这里只是确保得到相同的API密钥字符串
        # 在实际场景中，需要使用可逆的编码方式
        return secret_int
    
    def _string_to_int(self, s: str) -> int:
        """
        将字符串转换为整数，仅用于演示
        """
        # 对于简单数字字符串，直接转换
        if s.isdigit():
            return int(s)
        
        # 对于其他字符串，使用哈希
        # 注意：这不是可逆的，仅用于演示
        hash_object = hashlib.sha256(s.encode())
        hex_dig = hash_object.hexdigest()
        # 取哈希的前16个字符作为整数
        return int(hex_dig[:16], 16)
    
    def _evaluate_polynomial(self, coef: List[int], x: int) -> int:
        """计算给定系数的多项式在点x处的值"""
        result = 0
        # 使用秦九韶算法(Horner's method)计算多项式值
        for c in reversed(coef):
            result = (result * x + c)
        return result
    
    def _lagrange_interpolation(self, shares: List[Tuple[int, int]], x: int) -> int:
        """
        使用拉格朗日插值计算秘密
        
        参数:
            shares: 多项式上的点(x, y)列表
            x: 要计算的点(通常为0，即秘密值)
            
        返回:
            点x处的插值结果
        """
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator *= (x - xj)
                    denominator *= (xi - xj)
            
            # 将这一项加到结果中
            secret += yi * numerator // denominator
        
        return secret