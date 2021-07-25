def train(model, train_loader, test_loader, epochs=3, lr=1e-2, wd=0.0):
  optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
  model.train()
	min_loss = 999
	for i in range(epochs):
		stats = {'epoch' : i + 1, 'total' : epochs}
		print("Epoch %d"%(i+1))
		sum_loss = 0.0
		total = 0

		for features, labels in train_loader:
      batch = features.shape[0]
			y_pred = model(features)
			loss = F.mse_loss(y_pred, labels.view(-1, batch)[0].float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total += batch
			sum_loss += batch*loss.item()
		print("train loss %.3f" % (sum_loss/total))
		test_los = test_loss(model, test_loader)
		if test_los < min_loss:
			min_loss = test_los
			print("test loss: %.3f"%(test_los))
			no_improvements = 0
		else:
			no_improvements += 1

		history.append(stats)
		print(fmt.format(**stats))
		if no_improvements >= patience:
			break
      
def test_loss(model, test_loader):
    model.eval()
    sum_loss = 0.0
    min_loss = 999
    total = 0
    for features, labels in test_loader:
        batch = features.shape[0]
        y_pred = model(features)
        loss = F.mse_loss(y_pred, labels.view(-1, batch)[0].float())
        total += batch
        sum_loss += batch*loss.item()
    return sum_loss/total
  
