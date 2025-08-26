"""
PyTorch implementation of XRD/PDF neural networks for phase identification.

This module provides a complete replacement for the TensorFlow-based implementation,
offering improved performance, better uncertainty estimation, and enhanced training capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from random import shuffle
from typing import Tuple, List, Optional
import sys
import os
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)


class AlwaysDropout(nn.Module):
    """
    Custom dropout layer that applies dropout during both training and inference.
    This is essential for Monte Carlo dropout uncertainty estimation.
    """
    
    def __init__(self, p: float = 0.5):
        super(AlwaysDropout, self).__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode."""
        return F.dropout(x, p=self.p, training=True)


class XRDNet(nn.Module):
    """
    1D Convolutional Neural Network for XRD pattern classification.
    
    Optimized architecture for XRD analysis with progressive kernel size reduction
    and custom dropout for uncertainty estimation.
    """
    
    def __init__(self, num_classes: int, n_dense: List[int] = [3100, 1200], 
                 dropout_rate: float = 0.7):
        """
        Args:
            num_classes: Number of reference phases (output classes)
            n_dense: List containing sizes of dense layers [layer1_size, layer2_size]
            dropout_rate: Dropout probability for regularization
        """
        super(XRDNet, self).__init__()
        
        # Convolutional layers with progressive kernel size reduction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=1, padding=35//2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=30, stride=1, padding=30//2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=25, stride=1, padding=25//2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=20, stride=1, padding=20//2)
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        self.conv5 = nn.Conv1d(64, 64, kernel_size=15, stride=1, padding=15//2)
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=10//2)
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        # Calculate flattened size after all conv and pooling layers
        self._calculate_conv_output_size()
        
        # Dense layers with dropout and batch normalization
        self.dropout1 = AlwaysDropout(dropout_rate)
        self.fc1 = nn.Linear(self.conv_output_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])
        
        self.dropout2 = AlwaysDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])
        
        self.dropout3 = AlwaysDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the output size after all convolutional and pooling layers."""
        # Start with input size of 4501
        size = 4501
        
        # Apply each conv + pool operation
        size = ((size + 2*1 - 3) // 2) + 1  # pool1
        size = ((size + 2*1 - 3) // 2) + 1  # pool2  
        size = ((size + 2*0 - 2) // 2) + 1  # pool3
        size = ((size + 2*0 - 1) // 2) + 1  # pool4
        size = ((size + 2*0 - 1) // 2) + 1  # pool5
        size = ((size + 2*0 - 1) // 2) + 1  # pool6
        
        self.conv_output_size = size * 64  # 64 channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4501, 1) or (batch_size, 1, 4501)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Ensure correct input shape: (batch_size, channels, length)
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)  # (batch_size, 4501, 1) -> (batch_size, 1, 4501)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 4501) -> (batch_size, 1, 4501)
            
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        
        # Flatten for dense layers
        x = x.view(x.size(0), -1)
        
        # Dense layers with dropout and batch normalization
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)


class PDFNet(nn.Module):
    """
    Simplified 1D CNN optimized for PDF (Pair Distribution Function) analysis.
    
    Uses a single large convolutional kernel followed by aggressive pooling,
    designed specifically for PDF pattern characteristics.
    """
    
    def __init__(self, num_classes: int, n_dense: List[int] = [3100, 1200], 
                 dropout_rate: float = 0.7):
        """
        Args:
            num_classes: Number of reference phases (output classes)
            n_dense: List containing sizes of dense layers [layer1_size, layer2_size]
            dropout_rate: Dropout probability for regularization
        """
        super(PDFNet, self).__init__()
        
        # Single convolutional layer with large kernel
        self.conv1 = nn.Conv1d(1, 64, kernel_size=60, stride=1, padding='same')
        
        # Aggressive pooling sequence
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        # Calculate flattened size
        self._calculate_conv_output_size()
        
        # Dense layers
        self.dropout1 = AlwaysDropout(dropout_rate)
        self.fc1 = nn.Linear(self.conv_output_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])
        
        self.dropout2 = AlwaysDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])
        
        self.dropout3 = AlwaysDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the output size after all pooling layers."""
        size = 4501
        size = ((size + 2*1 - 3) // 2) + 1  # pool1
        size = ((size + 2*1 - 3) // 2) + 1  # pool2
        size = ((size + 2*0 - 2) // 2) + 1  # pool3
        size = ((size + 2*0 - 1) // 2) + 1  # pool4
        size = ((size + 2*0 - 1) // 2) + 1  # pool5
        size = ((size + 2*0 - 1) // 2) + 1  # pool6
        self.conv_output_size = size * 64
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PDF network."""
        # Ensure correct input shape
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Convolutional layer
        x = F.relu(self.conv1(x))
        
        # Pooling sequence
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = self.pool6(x)
        
        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)


class XRDDataset(Dataset):
    """
    Custom PyTorch Dataset for XRD/PDF spectra data.
    
    Handles the conversion from the original nested numpy array format
    to a format suitable for PyTorch DataLoader.
    """
    
    def __init__(self, xrd_data: np.ndarray):
        """
        Args:
            xrd_data: Numpy array of shape (num_phases, num_spectra_per_phase, 4501, 1)
        """
        self.spectra = []
        self.labels = []
        
        # Flatten the nested structure and create labels
        for phase_idx, phase_spectra in enumerate(xrd_data):
            for spectrum in phase_spectra:
                # Convert to tensor and ensure correct shape
                spectrum_tensor = torch.FloatTensor(spectrum).squeeze()  # Remove extra dims
                if spectrum_tensor.dim() == 0:
                    spectrum_tensor = spectrum_tensor.unsqueeze(0)
                    
                self.spectra.append(spectrum_tensor)
                self.labels.append(phase_idx)
        
        self.spectra = torch.stack(self.spectra)
        self.labels = torch.LongTensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.labels[idx]


class DataSetUp(object):
    """
    Enhanced data setup class for PyTorch training.
    
    Provides train/test splitting and DataLoader creation with improved
    functionality compared to the original TensorFlow version.
    """
    
    def __init__(self, xrd: np.ndarray, testing_fraction: float = 0.0, 
                 batch_size: int = 32, num_workers: int = 4):
        """
        Args:
            xrd: Numpy array containing XRD spectra categorized by reference phase
            testing_fraction: Fraction of data to reserve for testing
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for data loading
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create train, validation, and optionally test DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
            test_loader is None if testing_fraction is 0
        """
        # Create dataset
        dataset = XRDDataset(self.xrd)
        
        if self.testing_fraction == 0:
            # Split into train (80%) and validation (20%)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, None
            
        else:
            # Split into train/val and test
            test_size = int(self.testing_fraction * len(dataset))
            trainval_size = len(dataset) - test_size
            
            trainval_dataset, test_dataset = random_split(
                dataset, [trainval_size, test_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            # Further split train/val
            train_size = int(0.8 * trainval_size)
            val_size = trainval_size - train_size
            
            train_dataset, val_dataset = random_split(
                trainval_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, test_loader


def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                num_phases: int, num_epochs: int, is_pdf: bool, 
                n_dense: List[int] = [3100, 1200], dropout_rate: float = 0.7,
                learning_rate: float = 0.001, device: str = None) -> nn.Module:
    """
    Train the neural network model with enhanced features.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        num_phases: Number of reference phases
        num_epochs: Number of training epochs
        is_pdf: Whether to use PDF-optimized architecture
        n_dense: Dense layer sizes
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer
        device: Device to use for training ('cuda' or 'cpu')
        
    Returns:
        Trained model
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    print(f"Training on device: {device}")
    
    # Create model
    if is_pdf:
        model = PDFNet(num_phases, n_dense, dropout_rate)
        print("Using PDF-optimized architecture")
    else:
        model = XRDNet(num_phases, n_dense, dropout_rate)
        print("Using XRD-optimized architecture")
        
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with progress tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                epoch_val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")
    
    print("Training completed!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    
    return model


def save_model(model: nn.Module, filepath: str, is_pdf: bool = False):
    """
    Save the trained model with metadata.
    
    Args:
        model: Trained PyTorch model
        filepath: Path to save the model
        is_pdf: Whether this is a PDF model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'PDFNet' if is_pdf else 'XRDNet',
        'num_classes': model.fc3.out_features,
        'n_dense': [model.fc1.out_features, model.fc2.out_features],
        'dropout_rate': model.dropout1.p,
        'is_pdf': is_pdf
    }, filepath)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath: str, device: str = None) -> nn.Module:
    """
    Load a saved PyTorch model.
    
    Args:
        filepath: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model based on saved type
    if checkpoint['is_pdf']:
        model = PDFNet(
            checkpoint['num_classes'],
            checkpoint['n_dense'],
            checkpoint['dropout_rate']
        )
    else:
        model = XRDNet(
            checkpoint['num_classes'],
            checkpoint['n_dense'],
            checkpoint['dropout_rate']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from: {filepath}")
    print(f"Model type: {checkpoint['model_type']}")
    print(f"Number of classes: {checkpoint['num_classes']}")
    
    return model


def main(xrd: np.ndarray, num_epochs: int, testing_fraction: float, 
         is_pdf: bool, fmodel: str = 'Model.pth'):
    """
    Main training function with enhanced PyTorch implementation.
    
    Args:
        xrd: XRD/PDF spectra data
        num_epochs: Number of training epochs
        testing_fraction: Fraction of data for testing
        is_pdf: Whether to use PDF-optimized architecture
        fmodel: Filename to save the trained model
    """
    print("Setting up data...")
    data_setup = DataSetUp(xrd, testing_fraction)
    num_phases = data_setup.num_phases
    
    print(f"Number of phases: {num_phases}")
    print(f"Dataset size: {len(XRDDataset(xrd))} spectra")
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_setup.get_dataloaders()
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Train model
    model = train_model(train_loader, val_loader, num_phases, num_epochs, is_pdf)
    
    # Save model
    save_model(model, fmodel, is_pdf)
    
    # Test model if test data is available
    if test_loader is not None:
        from .pytorch_models import test_model
        test_accuracy = test_model(model, test_loader)
        print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    return model


def test_model(model: nn.Module, test_loader: DataLoader, device: str = None) -> float:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to use for testing
        
    Returns:
        Test accuracy as percentage
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            test_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    test_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Test Results: Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy