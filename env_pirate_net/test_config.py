#!/usr/bin/env python3
"""
Test configuration system with different scenarios
"""

from settings import Config, create_reynolds_configs

def test_default_config():
    """Test default configuration"""
    print("Testing default configuration...")
    config = Config()
    print(config.summary())
    assert config.network.out_dim == 3
    assert config.physics.reynolds_number == 100.0
    print("✓ Default config test passed\n")

def test_custom_config():
    """Test custom configuration"""
    print("Testing custom configuration...")
    config = Config()
    config.physics.reynolds_number = 1000.0
    config.training.epochs = 20000
    config.network.hidden_dim = 512
    
    print(f"Custom Re: {config.physics.reynolds_number}")
    print(f"Custom epochs: {config.training.epochs}")
    print(f"Custom hidden_dim: {config.network.hidden_dim}")
    print("✓ Custom config test passed\n")

def test_reynolds_configs():
    """Test multiple Reynolds number configurations"""
    print("Testing Reynolds number configurations...")
    re_values = [100, 400, 1000]
    configs = create_reynolds_configs(re_values)
    
    for config in configs:
        print(f"Re={config.physics.reynolds_number}: epochs={config.training.epochs}, name='{config.experiment_name}'")
    
    assert len(configs) == 3
    assert configs[2].training.epochs == 20000  # Re=1000 should have more epochs
    print("✓ Reynolds configs test passed\n")

def test_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    try:
        config = Config()
        config.network.out_dim = 2  # Should be 3 for (u,v,p)
        config._validate_config()
        assert False, "Should have raised assertion error"
    except AssertionError:
        print("✓ Validation correctly caught invalid out_dim")
    
    try:
        config = Config()
        config.physics.reynolds_number = -1  # Should be positive
        config._validate_config()  
        assert False, "Should have raised assertion error"
    except AssertionError:
        print("✓ Validation correctly caught negative Reynolds number")
    
    print("✓ Validation tests passed\n")

if __name__ == "__main__":
    test_default_config()
    test_custom_config()
    test_reynolds_configs()
    test_validation()
    print("🎉 All configuration tests passed!")