import torch

def train_straight_flow(model, epochs, dataloader, optimizer):
    model.train()
    history = []
    for epoch in tqdm(range(1, epochs+1)):
        epoch_loss = 0.0
        num_batches = 0
        for x0_batch, x1_batch in dataloader:
            B = x0_batch.shape[0]
            t = torch.rand(B, 1).to(device)
            xt = (1-t) * x0_batch + t * x1_batch

            pred = model(xt, t)
            loss = F.mse_loss(pred, x1_batch - x0_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        mean_loss = epoch_loss / max(num_batches, 1)
        history.append(mean_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Mean Loss: {mean_loss:.4f}")
    return history


def batch_time_jacobian(y, t):
    # y: (B, D), t: (B, 1)
    grads = []
    for d in range(y.shape[1]):
        g = torch.autograd.grad(
            outputs=y[:, d].sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(g)
    return torch.cat(grads, dim=1)  # (B, D)


def distillate_flow_map_lagrangian(teacher, student, epochs, dataloader, optimizer):
    teacher.eval()
    student.train()
    history = []

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_loss = 0.0
        num_batches = 0

        for x0_batch, x1_batch in dataloader:
            B = x0_batch.shape[0]

            times = torch.rand(B, 2, device=device)
            times, _ = torch.sort(times, dim=1)
            s = times[:, 0:1]
            t = times[:, 1:2]

            z = torch.randn_like(x0_batch)
            I_s = interpolator.interpolate(x0_batch, x1_batch, z, s)

            t_req = t.clone().detach().requires_grad_(True)
            student_pred = student(I_s, s, t_req)

            dXdt = batch_time_jacobian(student_pred, t_req)

            with torch.no_grad():
                teacher_pred = teacher(student_pred.detach(), t_req)

            loss = F.mse_loss(dXdt, teacher_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        history.append(mean_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Mean Loss: {mean_loss:.4f}")

    return history


def progressive_flow_map_matching(teacher, student, epochs, K, dataloader, optimizer):
    teacher.eval()
    student.train()
    history = []
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_loss = 0.0
        num_batches = 0

        for x0_batch, x1_batch in dataloader:
            B = x0_batch.shape[0]

            times = torch.rand(B, 2, device=device)
            times, _ = torch.sort(times, dim=1)
            s = times[:, 0:1]
            t = times[:, -1:]

            z = torch.randn_like(x0_batch)
            I_s = interpolator.interpolate(x0_batch, x1_batch, z, s)

            pred_whole = student(I_s, s, t)
            
            t_prev = s
            pred_k = I_s
            for k in range(1, K + 1):
                t_k = s + k * (t - s) / K
                pred_k = teacher(pred_k, t_prev, t_k)
                t_prev = t_k
            
            loss = F.mse_loss(pred_k.detach(), pred_whole)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        history.append(mean_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Mean Loss: {mean_loss:.4f}")
    
    return history


def self_distillation_flow_map(model, epochs, dataloader, optimizer, device='cuda', eta=0.5):
    model.train()
    history = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_loss = 0.0
        num_batches = 0

        for x0_batch, x1_batch in dataloader:
            x0_batch, x1_batch = x0_batch.to(device), x1_batch.to(device)
            B = x0_batch.shape[0]
            
            # Batch split
            Md = int(eta * B)
            Mo = B - Md

            Lb = torch.tensor(0.0, device=device)
            Ld = torch.tensor(0.0, device=device)

            # Diagonal Loss
            if Md > 0:
                t_d = torch.rand(Md, 1, device=device)
                z_d = torch.randn_like(x0_batch[:Md])

                I_td, dt_Itd = otinterpolator.interpolate_both(x0_batch[:Md], x1_batch[:Md], z_d, t_d)
                
                velocity_self, w_d = model.velocity_and_w(I_td, t_d, t_d)

                mse_b = F.mse_loss(velocity_self, dt_Itd, reduction='none').view(Md, -1).mean(dim=1, keepdim=True)
                Lb = (torch.exp(-w_d) * mse_b + w_d).mean()

            # Distillation Loss
            if Mo > 0:
                times_o = torch.rand(Mo, 2, device=device)
                times_o, _ = torch.sort(times_o, dim=1)
                s_o = times_o[:, 0:1]
                t_o = times_o[:, -1:]
                
                z_o = torch.randn_like(x0_batch[Md:])
                I_so, _ = otinterpolator.interpolate_both(x0_batch[Md:], x1_batch[Md:], z_o, s_o)

                t_req = t_o.clone().detach().requires_grad_(True)
                
                pred = model(I_so, s_o, t_req) 
                
                velocity_pred, w_o = model.velocity_and_w(pred, t_req, t_req)

                dXdt = batch_time_jacobian(pred, t_req)

                mse_d = F.mse_loss(dXdt, velocity_pred.detach(), reduction='none').view(Mo, -1).mean(dim=1, keepdim=True)
                Ld = (torch.exp(-w_o) * mse_d + w_o).mean()

            loss = Lb + Ld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        history.append(mean_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Mean Loss: {mean_loss:.4f}")
    
    return history