# client.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import pad_collate
from utils import get_model_state_dict, load_model_state_dict
import copy

class Client:
    def __init__(self, client_id, model_architecture, local_dataset, config):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model_architecture).to(self.device)
        self.dataset = local_dataset
        self.config = config
        
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.config['batch_size'], shuffle=True,
            collate_fn=pad_collate, num_workers=4, pin_memory=True
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']
        )
        # self.optimizer = optim.Adam(
        #     self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']
        # )
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9, weight_decay=self.config['weight_decay']
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(self.data_loader) * self.config['local_epochs'],
            eta_min=1e-8
        )
        self.criterion = nn.CrossEntropyLoss()

        # 为 SCAFFOLD 初始化客户端控制变量
        self.control_variate_local = {
            name: torch.zeros_like(param) 
            for name, param in get_model_state_dict(self.model).items()
        }



    def set_weights(self, weights):
        load_model_state_dict(self.model, weights)

    def train(self, **kwargs):
        """
        执行本地训练。
        根据传入的kwargs，执行不同算法的训练逻辑。
        """
        # --- SCAFFOLD 特定的训练逻辑 ---
        if kwargs.get('algorithm') == 'scaffold':
            return self.train_scaffold(**kwargs)
        if kwargs.get('algorithm')  == 'scamoon':
            return self.train_scamoon(**kwargs)

        # --- FedAvg, MOON, FedProx 的通用训练逻辑 ---
        self.model.train()
        print(f"   >> 客户端 {self.client_id} 开始本地训练 ({self.config['local_epochs']} epochs)...")
        
        # 从kwargs获取策略特定的对象
        contrastive_criterion = kwargs.get('contrastive_criterion', None)
        global_model = kwargs.get('global_model', None)
        previous_model = kwargs.get('previous_model', None)
        mu = self.config.get('mu', 0)

        # FedProx 特定的参数
        prox_mu = kwargs.get('prox_mu', 0)
        global_model_weights = None
        if prox_mu > 0:
            global_model_weights = [param.detach().clone() for param in kwargs.get('global_model').parameters()]

        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits, rep_curr = self.model(inputs)
                
                # 1. 监督损失 (所有算法都需要)
                loss_sup = self.criterion(logits, labels)
                total_loss = loss_sup

                # 2. MOON 对比损失
                if contrastive_criterion and global_model and previous_model:
                    with torch.no_grad():
                        _, rep_pos = global_model(inputs)
                        _, rep_neg = previous_model(inputs)
                    
                    loss_con = contrastive_criterion(rep_curr, rep_pos.detach(), rep_neg.detach())
                    total_loss += mu * loss_con
                
                # 3. FedProx 近端项
                if prox_mu > 0 and global_model_weights is not None:
                    prox_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_model_weights):
                        prox_term += (local_param - global_param).norm(2)
                    total_loss += (prox_mu / 2) * prox_term

                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += total_loss.item() * inputs.size(0)
            avg_epoch_loss = epoch_loss / len(self.dataset)
            print(f"       - Client {self.client_id}, Epoch {epoch+1}/{self.config['local_epochs']}, Loss: {avg_epoch_loss:.4f}")
            
        
        return get_model_state_dict(self.model)

    def train_scaffold(self, **kwargs):
        """
        SCAFFOLD的专用本地训练函数 (根据新参考代码重构)
        核心变化: 不再修改梯度，而是在优化器更新后，直接修改模型权重。
        """
        print(f"   >> 客户端 {self.client_id} (SCAFFOLD) 开始本地训练...")
        self.model.train()
        
        control_variate_global = kwargs['control_variate_global']
        
        # 1. 保存初始状态
        # initial_weights用于计算ccv的更新，必须是深拷贝
        initial_weights = {k: v.clone() for k, v in get_model_state_dict(self.model).items()}
        # ccv_state是旧的客户端控制变量
        ccv_state = {k: v.clone() for k, v in self.control_variate_local.items()}
        # scv_state是服务器控制变量，需要放到客户端设备上
        scv_state = {k: v.to(self.device) for k, v in control_variate_global.items()}

        # 获取底层模型，以便遍历不带前缀的参数
        underlying_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # 2. 标准本地训练 (不再有梯度修正)
        num_steps = 0
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                logits = self.model(inputs, return_projection=False)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                # 【新逻辑】在优化器更新后，直接修正模型权重
                with torch.no_grad():
                    current_lr = self.optimizer.param_groups[0]['lr']
                    state_dict = get_model_state_dict(self.model)
                    for key in state_dict:
                        # state_dict[key] = state_dict[key] - lr * (c_global - c_local)
                        if key in scv_state:
                            state_dict[key].sub_( (scv_state[key] - ccv_state[key]) * current_lr )
                    load_model_state_dict(self.model, state_dict)
                
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                epoch_loss += loss.item() * inputs.size(0)
                num_steps += 1

            avg_epoch_loss = epoch_loss / len(self.dataset)
            print(f"       - Client {self.client_id} (SCAFFOLD), Epoch {epoch+1}/{self.config['local_epochs']}, Loss: {avg_epoch_loss:.4f}")

        # 3. 计算新的客户端控制变量 (ccv) 和其增量
        with torch.no_grad():
            final_weights = get_model_state_dict(self.model)
            local_lr = self.config['learning_rate'] # 使用基础学习率
            if num_steps > 0 and local_lr > 0:
                coeff = 1.0 / (num_steps * local_lr)
            else:
                coeff = 0

            # 更新本地 ccv
            # new_c_i = c_i - c + (x - y) / (K * lr)
            for name in ccv_state:
                 update_term = (initial_weights[name].to(self.device) - final_weights[name].to(self.device)) * coeff
                 self.control_variate_local[name].copy_(ccv_state[name] - scv_state[name] + update_term)

            # 计算 ccv 的增量: delta_c_i = new_c_i - c_i
            delta_ccv = {name: self.control_variate_local[name] - ccv_state[name] for name in ccv_state}

        # 4. 返回完整模型和 ccv 增量
        return final_weights, delta_ccv



    def train_scamoon(self, **kwargs):
        """
        ScaMoon的训练函数，结构与您成功的train_scaffold完全镜像。
        """
        print(f"   >> 客户端 {self.client_id} (ScaMoon) 开始本地训练...")
        self.model.train()
        
        # 1. 获取所有组件和参数
        control_variate_global = kwargs['control_variate_global']
        contrastive_criterion = kwargs['contrastive_criterion']
        global_model = kwargs['global_model']
        previous_model = kwargs['previous_model']
        mu = self.config.get('mu', 1.0) # mu由策略文件在预热期动态设置为0
        
        # 2. 保存初始状态
        initial_weights = {k: v.clone() for k, v in get_model_state_dict(self.model).items()}
        ccv_state = {k: v.clone() for k, v in self.control_variate_local.items()}
        scv_state = {k: v.to(self.device) for k, v in control_variate_global.items()}

        num_steps = 0
        
        # 3. 本地训练循环
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # a. 计算总损失 (包含MOON损失)
                logits, proj_curr = self.model(inputs)
                loss_sup = self.criterion(logits, labels)
                total_loss = loss_sup

                if mu > 0:
                    with torch.no_grad():
                        _, proj_pos = global_model(inputs)
                        _, proj_neg = previous_model(inputs)
                    loss_con = contrastive_criterion(proj_curr, proj_pos.detach(), proj_neg.detach())
                    total_loss += mu * loss_con
                
                # b. 标准优化器步骤
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # c. SCAFFOLD 事后修正
                with torch.no_grad():
                    current_lr = self.optimizer.param_groups[0]['lr']
                    state_dict = get_model_state_dict(self.model)
                    for key in state_dict:
                        if key in scv_state:
                            state_dict[key].sub_((scv_state[key] - ccv_state[key]) * current_lr)
                    load_model_state_dict(self.model, state_dict)

                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                
                epoch_loss += total_loss.item() * inputs.size(0)
                num_steps += 1

            avg_epoch_loss = epoch_loss / len(self.dataset)
            print(f"       - Client {self.client_id}, Epoch {epoch+1}/{self.config['local_epochs']}, Loss: {avg_epoch_loss:.4f}")

        # 4. 计算新的客户端控制变量
        with torch.no_grad():
            final_weights = get_model_state_dict(self.model)
            local_lr = self.config['learning_rate']
            if num_steps > 0 and local_lr > 0:
                coeff = 1.0 / (num_steps * local_lr)
            else:
                coeff = 0
            for name in ccv_state:
                 update_term = (initial_weights[name].to(self.device) - final_weights[name].to(self.device)) * coeff
                 self.control_variate_local[name].copy_(ccv_state[name] - scv_state[name] + update_term)

            delta_ccv = {name: self.control_variate_local[name] - ccv_state[name] for name in ccv_state}

        # 5. 返回完整模型和 ccv 增量
        return final_weights, delta_ccv