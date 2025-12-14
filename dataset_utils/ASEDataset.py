import ase.db
from torch_geometric.data import Data, Dataset, DataLoader
import torch
import numpy as np
import math

from typing import Optional, List, Dict, Any, Iterable, Union, Tuple
from abc import ABC, abstractmethod
import torch.nn as nn
from ase import Atoms
import os
from ase.db import connect
from . import schnetpack_properties as structure

class ASEDataset(Dataset):
    def __init__(self, db_path, orbital_basis, dtype=torch.float32, open_shell=False, start_idx=0, end_idx=None):

        print("Connecting to database...")
        self.db = ase.db.connect(db_path)
        print("connected.")
        total_rows = self.db.count()
        print(f"Total rows in database: {total_rows}")

        self.orbital_basis = orbital_basis
        self.open_shell = open_shell

        if end_idx is None:
            end_idx = total_rows

        if start_idx < 0 or end_idx > total_rows or start_idx >= end_idx:
            end_idx = total_rows
            print("Invalid start_idx or end_idx values (probably end_idx > num rows). Setting end_idx to total rows.")

        self.ids = []
        for i, row in enumerate(self.db.select(limit=end_idx - start_idx, offset=start_idx)):
            self.ids.append(row.id)

        self.dtype = dtype

    def _get_ids(self):
        """
        Not used, just required to have this function
        """
        ids = []
        for row in self.db.select():
            print("Getting row..", flush=True)
            ids.append(row.id)
            # limit number to read
            # if len(ids) > limit:
            #     break
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        # Get structure by id
        structure = self.db.get(self.ids[idx])

        # Extract atom positions and atomic numbers
        atoms = structure.toatoms()
        positions = atoms.positions
        atomic_numbers = atoms.numbers

        # Convert numpy arrays from structure.data to PyTorch tensors
        neighbour_list = structure.data['edge_index']
        edge_dist = torch.tensor(structure.data['edge_dist'], dtype=self.dtype)
        edge_index = torch.tensor(neighbour_list, dtype=torch.long)
        edge_mask = torch.tensor(structure.data['edge_mask']) if 'edge_mask' in structure.data else None
        reverse_edge_map = torch.tensor(structure.data['reverse_edge_map']) if 'reverse_edge_map' in structure.data else None

        # Targets
        node_labels = torch.tensor(structure.data['node_labels'], dtype=self.dtype)
        edge_labels = torch.tensor(structure.data['edge_labels'], dtype=self.dtype)
        energies = torch.tensor(structure.data['total_energy [Eh]'])
        # forces = torch.tensor(structure.data['gradient [Eh/bohr]'])
        # dipole = torch.tensor(structure.data['multipoles'][1])  # XX, YY, ZZ components
        # quadrupole = torch.tensor(structure.data['multipoles'][2])  # XY, XZ, YZ components

        # Legacy closed-shell databases (does not contain spin dimension, so we add it):
        if node_labels.ndim == 2:
            node_labels = node_labels.unsqueeze(0)
            edge_labels = edge_labels.unsqueeze(0)
            charge = 0
            spin_multiplicity = 1
        else:
            charge = structure.data['charge']
            spin_multiplicity = structure.data['spin_multiplicity']

        # Handle individual closed-shell molecules in open-shell training by setting alphafock==betafock:
        if self.open_shell and node_labels.ndim == 3 and node_labels.shape[0] == 1:
            print("[Adding 2nd spin dimension to closed shell molecule for open shell training]")
            node_labels = node_labels.repeat(2, 1, 1)
            edge_labels = edge_labels.repeat(2, 1, 1)

        # metadata:
        folder_name = structure.data['folder_name']

        # Create PyTorch Geometric Data object with alpha/beta targets if open shell
        # NOTE: the alpha and beta targets are seperated to make collation easier
        if not self.open_shell:
            data = Data(
                pos=torch.tensor(positions, dtype=self.dtype),
                x=torch.tensor(atomic_numbers),
                edge_index=edge_index,
                edge_attr=edge_dist,
                edge_mask=edge_mask,
                reverse_edge_map=reverse_edge_map,
                y=edge_labels[0],
                node_y=node_labels[0],
                atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
                nedges=len(edge_index[0]),
                natoms=len(atomic_numbers),
                energies=energies,
                num_atoms_in_molecule=len(atomic_numbers),
                charge=charge,
                spin_multiplicity=int(spin_multiplicity),
                folder_name=folder_name,
            )
        else:
            data = Data(
                pos=torch.tensor(positions, dtype=self.dtype),
                x=torch.tensor(atomic_numbers),
                edge_index=edge_index,
                edge_attr=edge_dist,
                edge_mask=edge_mask,
                reverse_edge_map=reverse_edge_map,
                y_alpha=edge_labels[0],
                y_beta=edge_labels[1],
                node_y_alpha=node_labels[0],
                node_y_beta=node_labels[1],
                atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
                nedges=len(edge_index[0]),
                natoms=len(atomic_numbers),
                energies=energies,
                num_atoms_in_molecule=len(atomic_numbers),
                charge=charge,
                spin_multiplicity=int(spin_multiplicity),
                folder_name=folder_name,
            )

        # Store orbital basis (dictionary)
        data.orbital_basis = self.orbital_basis

        return data

class sampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def sample_collate_fn(batch):
    return Batch.from_data_list(batch)

def distribute_data(base_folder, N_db_files, world_size, rank, N_global_train, N_global_val):
    """
    Calculates the file and index ranges for a single rank to load its portion
    of the global dataset, based on predefined total counts for training and validation.

    It assumes the first N_global_train molecules, followed by the next N_global_val 
    molecules, constitute the total dataset to be used.

    Args:
        base_folder (str): The common path prefix for the database files.
        N_db_files (int): Total number of database files (e.g., 16).
        world_size (int): Total number of training ranks (e.g., 64).
        rank (int): The current rank's index (0 to world_size - 1).
        N_global_train (int): The TOTAL number of molecules to be used for training.
        N_global_val (int): The TOTAL number of molecules to be used for validation.

    Returns:
        tuple: (train_entries, val_entries)
               Each entry is a list of dicts: {'db_file': str, 'start_idx': int, 'end_idx': int}
    """

    # --- Step 1: Calculate Global Index Map ---
    # This map records the global index start for each database file.

    global_map = []
    current_global_idx = 0
    total_molecules_available = 0

    for i in range(N_db_files):
        db_file = os.path.join(base_folder, f'omol_electrolytes_unsolvated_job_{i}.db')

        try:
            db = ase.db.connect(db_file)
            count = db.count()
        except Exception as e:
            print(f"Warning: Could not open {db_file}. Skipping. Error: {e}")
            count = 0

        if count > 0:
            global_map.append({
                'db_file': db_file,
                'total_count': count,
                'global_start': current_global_idx
            })
            current_global_idx += count
            total_molecules_available += count
    
    # Safety Check
    if N_global_train + N_global_val > total_molecules_available:
        raise ValueError(
            f"Requested total molecules ({N_global_train + N_global_val}) exceeds "
            f"total available molecules ({total_molecules_available})."
        )
    
    N_global_total_used = N_global_train + N_global_val

    # --- Step 2: Determine Rank's Global Range for Training and Validation ---

    # Training Range (Indices [0, N_global_train) of the used dataset)
    N_train_per_rank = math.ceil(N_global_train / world_size)
    rank_train_start = rank * N_train_per_rank
    rank_train_end = min(rank_train_start + N_train_per_rank, N_global_train)

    # Validation Range (Indices [N_global_train, N_global_total_used) of the used dataset)
    N_val_per_rank = math.ceil(N_global_val / world_size)
    rank_val_start = N_global_train + rank * N_val_per_rank
    rank_val_end = min(rank_val_start + N_val_per_rank, N_global_total_used)

    # --- Step 3: Map Global Indices to Local DB Files (Training & Validation) ---

    train_entries = []
    val_entries = []

    # Iterate through the global map to find all necessary segments
    for entry in global_map:
        db_file = entry['db_file']
        db_global_start = entry['global_start']
        db_global_end = db_global_start + entry['total_count']

        # ----------------------------------------
        # A. Training Split Mapping
        # The global training set occupies the indices [0, N_global_train)
        
        # Segment of the DB file that belongs to the global training set
        db_train_end_limit = min(db_global_end, N_global_train)
        
        # Check for overlap between rank's training range and DB file's training segment
        # The rank's range is relative to the start of the global training set (index 0)
        # We need to shift the rank's range into the overall global index space [0, total_molecules_available)
        
        global_train_overlap_start = max(rank_train_start, db_global_start)
        global_train_overlap_end = min(rank_train_end, db_train_end_limit)

        if global_train_overlap_start < global_train_overlap_end:
            # Calculate local indices (offset from the file's start, row 0)
            local_start_idx = global_train_overlap_start - db_global_start
            local_end_idx = local_start_idx + (global_train_overlap_end - global_train_overlap_start)

            train_entries.append({
                'db_file': db_file,
                'start_idx': local_start_idx,
                'end_idx': local_end_idx
            })

        # ----------------------------------------
        # B. Validation Split Mapping
        # The global validation set occupies the indices [N_global_train, N_global_total_used)
        
        # Segment of the DB file that belongs to the global validation set
        db_val_start_limit = max(db_global_start, N_global_train)
        db_val_end_limit = min(db_global_end, N_global_total_used)
        
        # Check for overlap between rank's validation range and DB file's validation segment
        global_val_overlap_start = max(rank_val_start, db_val_start_limit)
        global_val_overlap_end = min(rank_val_end, db_val_end_limit)

        if global_val_overlap_start < global_val_overlap_end:
            # Calculate local indices
            local_start_idx = global_val_overlap_start - db_global_start
            local_end_idx = local_start_idx + (global_val_overlap_end - global_val_overlap_start)

            val_entries.append({
                'db_file': db_file,
                'start_idx': local_start_idx,
                'end_idx': local_end_idx
            })

    return train_entries, val_entries

# The following is directly copied from [https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/data/atoms.py#L36] to avoid installing schnetpack
class Transform(nn.Module):
    """
    Base class for all transforms.
    The base class ensures that the reference to the data and datamodule attributes are
    initialized.
    Transforms can be used as pre- or post-processing layers.
    They can also be used for other parts of a model, that need to be
    initialized based on data.

    To implement a new transform, override the forward method. Preprocessors are applied
    to single examples, while postprocessors operate on batches. All transforms should
    return a modified `inputs` dictionary.

    """

    def datamodule(self, value):
        """
        Extract all required information from data module automatically when using
        PyTorch Lightning integration. The transform should also implement a way to
        set these things manually, to make it usable independent of PL.

        Do not store the datamodule, as this does not work with torchscript conversion!
        """
        pass

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def teardown(self):
        pass

class BaseAtomsData(ABC):
    """
    Base mixin class for atomistic data. Use together with PyTorch Dataset or
    IterableDataset to implement concrete data formats.
    """

    def __init__(
        self,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[Transform]] = None,
        subset_idx: Optional[List[int]] = None,
    ):
        """
        Args:
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing transforms (see schnetpack.data.transforms)
            subset: List of data indices.
        """
        self._transform_module = None
        self.load_properties = load_properties
        self.load_structure = load_structure
        self.transforms = transforms
        self.subset_idx = subset_idx

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value: Optional[List[Transform]]):
        self._transforms = []
        self._transform_module = None

        if value is not None:
            for tf in value:
                self._transforms.append(tf)
            self._transform_module = torch.nn.Sequential(*self._transforms)

    def subset(self, subset_idx: List[int]):
        assert (
            subset_idx is not None
        ), "Indices for creation of the subset need to be provided!"
        ds = copy.copy(self)
        if ds.subset_idx:
            ds.subset_idx = [ds.subset_idx[i] for i in subset_idx]
        else:
            ds.subset_idx = subset_idx
        return ds

    @property
    @abstractmethod
    def available_properties(self) -> List[str]:
        """Available properties in the dataset"""
        pass

    @property
    @abstractmethod
    def units(self) -> Dict[str, str]:
        """Property to unit dict"""
        pass

    @property
    def load_properties(self) -> List[str]:
        """Properties to be loaded"""
        if self._load_properties is None:
            return self.available_properties
        else:
            return self._load_properties

    @load_properties.setter
    def load_properties(self, val: List[str]):
        if val is not None:
            props = self.available_properties
            assert all(
                [p in props for p in val]
            ), "Not all given properties are available in the dataset!"
        self._load_properties = val

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""
        pass

    @property
    @abstractmethod
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        """Single-atom reference values for properties"""
        pass

    @abstractmethod
    def update_metadata(self, **kwargs):
        pass

    @abstractmethod
    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        pass

    @staticmethod
    @abstractmethod
    def create(
        datapath: str,
        position_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Dict[str, List[float]],
        **kwargs,
    ) -> "BaseAtomsData":
        pass

    @abstractmethod
    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        key_value_list: Optional[List[Dict[str, Any]]] = None,
    ):
        pass

    @abstractmethod
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        pass


class ASEAtomsData(BaseAtomsData):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database.

    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_structure: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            subset_idx: List of data indices.
            units: property-> unit string dictionary that overwrites the native units
                of the dataset. Units are converted automatically during loading.
        """
        self.datapath = datapath

        BaseAtomsData.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
        )

        self._check_db()
        self.conn = connect(self.datapath, use_lock_file=False)

        # initialize units
        md = self.metadata
        if "_distance_unit" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a distance unit set. Please add units to the "
                + "dataset using `spkconvert`!"
            )
        if "_property_unit_dict" not in md.keys():
            raise AtomsDataError(
                "Dataset does not have a property units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )

        if distance_unit:
            self.distance_conversion = spk.units.convert_units(
                md["_distance_unit"], distance_unit
            )
            self.distance_unit = distance_unit
        else:
            self.distance_conversion = 1.0
            self.distance_unit = md["_distance_unit"]

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = spk.units.convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        return self.conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            self.conn, idx, self.load_properties, self.load_structure
        )
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataError(f"ASE DB does not exists at {self.datapath}")

        if self.subset_idx:
            with connect(self.datapath, use_lock_file=False) as conn:
                n_structures = conn.count()

            assert max(self.subset_idx) < n_structures

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        """
        Return property dictionary at given indices.

        Args:
            indices: data indices
            load_properties (sequence or None): subset of available properties to load
            load_structure: load and return structure

        Returns:
            properties (dict): dictionary with molecular properties

        """
        if load_properties is None:
            load_properties = self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif type(indices) is int:
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif type(indices) is int:
                indices = [indices]

        # read from ase db
        for i in indices:
            yield self._get_properties(
                self.conn,
                i,
                load_properties=load_properties,
                load_structure=load_structure,
            )

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(row["pbc"])

        return properties

    # Metadata

    @property
    def metadata(self):
        return self.conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        self.conn.metadata = val

    def update_metadata(self, **kwargs):
        assert all(
            key[0] != 0 for key in kwargs
        ), "Metadata keys starting with '_' are protected!"

        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        """Dictionary of properties to units"""
        return self._units

    @property
    def atomrefs(self) -> Dict[str, torch.Tensor]:
        md = self.metadata
        arefs = md["atomrefs"]
        arefs = {k: self.conversions[k] * torch.tensor(v) for k, v in arefs.items()}
        return arefs

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEAtomsData":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
                reference values of the property. This is especially useful for
                extensive properties such as the energy, where the single atom energies
                contribute a major part to the overall value.
            kwargs: Pass arguments to init.

        Returns:
            newly created ASEAtomsData

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        atomrefs = atomrefs or {}

        with connect(datapath) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
                "atomrefs": atomrefs,
            }

        return ASEAtomsData(datapath, **kwargs)

    # add systems
    def add_system(self, atoms: Optional[Atoms] = None, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        self._add_system(self.conn, atoms, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
        key_value_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
            key_value_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        if key_value_list is None:
            key_value_list = [{}] * len(property_list)

        for at, prop, key_val in zip(atoms_list, property_list, key_value_list):
            self._add_system(
                self.conn,
                at,
                key_val,
                **prop,
            )

    def _add_system(
        self,
        conn,
        atoms: Optional[Atoms] = None,
        key_val: Optional[Dict[str, Any]] = None,
        **properties,
    ):
        """Add systems to DB"""
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        # add available properties to database
        valid_props = set().union(
            conn.metadata["_property_unit_dict"].keys(),
            [structure.Z, structure.R, structure.cell, structure.pbc],
        )
        for prop in properties:
            if prop not in valid_props:
                logger.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )
        for key in key_val:
            if key not in valid_props:
                logger.warning(
                    f"Property `{key}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )

        data = {}
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                if pname in properties:
                    data[pname] = properties[pname]
                if pname in key_val:
                    data[pname] = key_val[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)
