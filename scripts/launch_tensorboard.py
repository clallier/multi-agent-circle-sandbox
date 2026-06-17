import os
import glob
import sys
import shutil

class TensorBoardLauncher:
    """Manages the discovery, setup of symlinks, and launching of TensorBoard.
    
    This class scans the workspace for training directories, creates a clean
    temporary folder containing symbolic links to the TensorBoard subdirectories,
    and then starts TensorBoard pointing to this temporary folder. This prevents
    TensorBoard from traversing unrelated large directories like .venv/ and
    resolves issue with complex multi-logdir configurations.
    
    Attributes:
        _pattern (str): The glob pattern to search for run directories.
        _logs_dir (str): The path to the directory that will hold the symlinks.
    """
    
    def __init__(self, pattern: str = "./test_circle_sandbox_*", logs_dir: str = "./tb_logs"):
        """Initializes the TensorBoardLauncher with discovery settings.
        
        Args:
            pattern (str): Glob pattern to find run directories. Defaults to "./test_circle_sandbox_*".
            logs_dir (str): Destination directory for unified symlinks. Defaults to "./tb_logs".
            
        Returns:
            None.
            
        Raises:
            None.
            
        Example:
            launcher = TensorBoardLauncher(pattern="./test_*", logs_dir="./tb_logs")
        """
        self._pattern = pattern
        self._logs_dir = os.path.abspath(logs_dir)

    def find_run_dirs(self) -> list:
        """Finds all run directories matching the configured pattern.
        
        It scans the filesystem, sorts directories, and converts them to absolute paths.
        
        Args:
            None.
            
        Returns:
            list[str]: Sorted list of absolute directory paths matching the pattern.
            
        Raises:
            FileNotFoundError: If no directories match the pattern.
            
        Example:
            dirs = launcher.find_run_dirs()
        """
        dirs = sorted(glob.glob(self._pattern))
        if not dirs:
            raise FileNotFoundError(f"No directories found matching pattern: {self._pattern}")
        return [os.path.abspath(d) for d in dirs]

    def recreate_clean_dir(self) -> None:
        """Cleans and recreates the target symlinks directory.
        
        This deletes any existing directory, symlink, or file at the path and
        recreates it as a clean directory to prevent stale links from prior runs.
        
        Args:
            None.
            
        Returns:
            None.
            
        Raises:
            OSError: If deletion or directory creation fails.
            
        Example:
            launcher.recreate_clean_dir()
        """
        if os.path.exists(self._logs_dir):
            if os.path.islink(self._logs_dir):
                os.unlink(self._logs_dir)
            elif os.path.isdir(self._logs_dir):
                shutil.rmtree(self._logs_dir)
            else:
                os.remove(self._logs_dir)
        os.makedirs(self._logs_dir, exist_ok=True)

    def find_latest_event_file(self, tb_dir: str) -> str:
        """Finds the latest TensorBoard event file in a directory.
        
        It looks for files matching the events.out.tfevents.* pattern and selects
        the one with the latest modification time.
        
        Args:
            tb_dir (str): Path to the directory containing event files.
            
        Returns:
            str: Absolute path to the latest event file.
            
        Raises:
            FileNotFoundError: If no event files are found.
            
        Example:
            latest = launcher.find_latest_event_file("/path/to/tb")
        """
        pattern = os.path.join(tb_dir, "events.out.tfevents.*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No event files found in {tb_dir}")
        files.sort(key=os.path.getmtime)
        return os.path.abspath(files[-1])

    def link_run(self, run_dir: str) -> None:
        """Links the latest run's event file into a nested logs directory.
        
        It finds the latest event file in the run's tensorboard directory,
        creates a subfolder in tb_logs/ named after the run, and symlinks
        only that latest event file into it.
        
        Args:
            run_dir (str): Absolute path to the run directory to be linked.
            
        Returns:
            None.
            
        Raises:
            OSError: If creating the directory or symlink fails.
            
        Example:
            launcher.link_run("/path/to/run_1")
        """
        name = os.path.basename(run_dir)
        tb_dir = os.path.join(run_dir, "tensorboard")
        target_tb_dir = tb_dir if os.path.isdir(tb_dir) else run_dir
        
        try:
            latest_file = self.find_latest_event_file(target_tb_dir)
            dest_dir = os.path.join(self._logs_dir, name)
            os.makedirs(dest_dir, exist_ok=True)
            link_path = os.path.join(dest_dir, os.path.basename(latest_file))
            os.symlink(latest_file, link_path)
        except FileNotFoundError:
            # Fall back to symlinking the entire directory if no tfevents files are found
            link_path = os.path.join(self._logs_dir, name)
            os.symlink(target_tb_dir, link_path)

    def prepare_logs(self) -> str:
        """Finds all runs, recreates logs directory, and symlinks all runs.
        
        It orchestrates the discovery and creation of symlinks for all matching runs.
        
        Args:
            None.
            
        Returns:
            str: Absolute path to the prepared logs directory containing all symlinks.
            
        Raises:
            FileNotFoundError: If no run directories are found.
            OSError: If filesystem operations fail.
            
        Example:
            logs_path = launcher.prepare_logs()
        """
        run_dirs = self.find_run_dirs()
        self.recreate_clean_dir()
        for run_dir in run_dirs:
            self.link_run(run_dir)
        return self._logs_dir

    def launch(self) -> None:
        """Orchestrates log preparation and executes TensorBoard.
        
        This method prepares the logs directory and uses os.execvp to replace the
        current python process with the running TensorBoard server, passing along any
        additional arguments.
        
        Args:
            None.
            
        Returns:
            None. Does not return upon successful execution of os.execvp.
            
        Raises:
            SystemExit: If log preparation fails.
            
        Example:
            launcher.launch()
        """
        try:
            logs_path = self.prepare_logs()
            print(f"Launching TensorBoard pointing to: {logs_path}")
            os.execvp("tensorboard", ["tensorboard", "--logdir", logs_path] + sys.argv[1:])
        except (FileNotFoundError, OSError) as err:
            print(f"Error preparing TensorBoard: {err}", file=sys.stderr)
            sys.exit(1)

def main() -> None:
    """Main entrypoint for running the TensorBoard launcher.
    
    Creates a TensorBoardLauncher instance and runs the launch process.
    
    Args:
        None.
        
    Returns:
        None.
        
    Raises:
        None.
        
    Example:
        main()
    """
    launcher = TensorBoardLauncher()
    launcher.launch()

if __name__ == "__main__":
    main()

